''' Contains tools to train our model using the mirrored distribution strategy.


Notes
-----
This module can be run as a script. To do so, just type 'python train.py' in the terminal. The help 
page should contain all the options you might possibly need.

This module contains the actual training function used to train a model
using the built-in Keras model.fit API, or alternatively a custom training loop. 
It combines the data input pipeline defined in data_input.py (which makes use of 
the .tfrecord files previously generated by preprocessing.py), 
the CNN architecture proposed by Pon et. al (2018) defined in orpheus_model.py, 
and a standard training loop with mirrored strategy 
integrated for multiple-GPU training.

The training function tries to optimise BinaryCrossentropy for each tag
prediction (batch loss is summed over the batch size, instead of being averaged), and 
displays the area under the ROC and PR curves as metrics. The optimizer can
be fully specified in the config.json file.

The learning rate can be halved whenever the validation AUC PR does not improve
for more than plateau_patience epochs. Moreover, training can be automatically
interrupted if the validation AUC PR does not improve for more than early_stop_patience
epochs. In these two cases, the delta which determines what is to be consiered a valid improvement
is determined by plateau_min_d and early_stop_min_d respectively.

Logs and checkpoints are automatically saved in subdirectories named after the 
frontend adopted and a timestamp. 
Logs can be accessed and viewed using TensorBoard. 

The config.json file is automatically copied in the directory for future reference.

IMPORTANT: if trying to profile a batch in TensorBoard, make sure the environment
variable LD_LIBRARY_PATH is specified.
(e.g. 'export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:
                               /usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64"')
'''

import argparse
import datetime
import json
import math
import os
import shutil

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from data_input import generate_datasets_from_dir
from orpheus_model import build_model
from orpheus_model import parse_config_json

class Learner:
    def __init__(self, frontend, train_dataset, valid_dataset, strategy, config, restore=None, custom_loop=True):
        # initialize training variables and strategy
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.frontend = frontend
        self.config = config
        self.strategy = strategy

        # initialize timestamp 
        self.timestamp = restore or datetime.datetime.now().strftime("%d%m%y-%H%M") # if restoring from previously saved checkpoint, use the 'old' timestamp

        # initialize directory to save logs and checkpoints in (create it, if necessary)
        self.log_dir = os.path.join(os.path.expanduser(self.config.log_dir), frontend[:13] + '_' + self.timestamp)
        
        if not restore:
            if not os.path.isdir(self.log_dir):
                os.makedirs(self.log_dir)
                shutil.copy(self.config.path, self.log_dir) # copy config.json in the same folder where logs and checkpoints will be saved

        # initialize model, loss, metrics, optimizer
        with self.strategy.scope():
            self.model = build_model(frontend, num_output_neurons=self.config.num_output_neurons, num_dense_units=self.config.num_dense_units, y_input=self.config.melspect_y)

            self.optimizer = tf.keras.optimizers.get({"class_name": self.config.optimizer_name, "config": self.config.optimizer})

            self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM)

            self.metric_1 = tf.keras.metrics.AUC(curve='ROC',
                                                       name='train_ROC-AUC', 
                                                       dtype=tf.float32)
            self.metric_2 = tf.keras.metrics.AUC(curve='PR',
                                                       name='train_PR-AUC',
                                                       dtype=tf.float32)

            if custom_loop:
                # initialize (and restore) checkpoint
                self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
            
                if restore:
                    file = tf.train.latest_checkpoint(self.log_dir)
                    if file:
                        self.checkpoint.restore(file).assert_consumed()
                    else:
                        raise FileNotFoundError
                
                # initialize tensorboard summary writers
                self.train_log_dir = os.path.join(self.log_dir, 'train/')
                self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
                self.valid_log_dir = os.path.join(self.log_dir, 'validation/')
                self.valid_summary_writer = tf.summary.create_file_writer(self.valid_log_dir)
                self.profiler_log_dir = os.path.join(self.log_dir, 'profile/')
                self.profiler_summary_writer = tf.summary.create_file_writer(self.profiler_log_dir)

    def train_with_fit(self, epochs, steps_per_epoch=None, restore=None, update_freq=1):
        ''' Creates a compiled instance of the training model and trains it for 'epochs' epochs.

        Parameters
        ----------
        train_dataset: tf.data.Dataset
            The training dataset.
            
        valid_dataset: tf.data.Dataset
            The validation dataset. If None, validation will be disabled. Tfe callbacks might not work properly.

        frontend: {'waveform', 'log-mel-spectrogram'}
            The frontend to adopt.
            
        strategy: tf.distribute.Strategy
            Strategy for multi-GPU distribution.

        config: argparse.Namespace
            Instance of the config namespace. It is generated when parsing the config.json file.
            
        epochs: int
            Specifies the number of epochs to train for.

        steps_per_epoch: int
            Specifies the number of steps to perform for each epoch. If None, the whole dataset will be used.
        
        restore: str
            Specifies the timestamp of the checkpoint to restore. Should be a timestamp in the 'YYMMDD-hhmm' format.

        update_freq: int
            Specifies the number of batches to wait before writing to logs. Note that writing too frequently can slow down training.

        profile_batch: int
            Specifies which batch to profile. Set to 0 to disable.
        '''

        with self.strategy.scope():
            # compile the model
            self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[[tf.keras.metrics.AUC(curve='ROC', name='ROC-AUC'), tf.keras.metrics.AUC(curve='PR', name='PR-AUC')]])
            
            # restore the model (if restore timestamp is provided)
            if restore:
                self.model.load_weights(os.path.join(os.path.expanduser(self.config.log_dir), self.frontend + '_' + restore))

        # initialize training callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath = os.path.join(self.log_dir, 'mymodel.h5'),
                monitor = 'val_PR-AUC',
                mode = 'max',
                save_best_only = True,
                save_freq = 'epoch',
                verbose = 1,
            ),

            tf.keras.callbacks.TensorBoard(
                log_dir = self.log_dir,
                histogram_freq = 1,
                write_graph = False,
                update_freq = update_freq,
                profile_batch = 0,
            ),

            tf.keras.callbacks.TerminateOnNaN(),
        ]

        if self.config.early_stop_patience is not None:

            self.config.early_stop_min_delta = self.config.early_stop_min_delta or 0

            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor = 'val_PR-AUC',
                    mode = 'max',
                    min_delta = self.config.early_stop_min_delta,
                    restore_best_weights = True,
                    patience = self.config.early_stop_patience,
                    verbose = 1,
                ),
            )
        
        if self.config.reduceLRoP_patience is not None:

            self.config.reduceLRoP_factor = self.config.reduceLRoP_factor or 0.5
            self.config.reduceLRoP_min_delta = self.config.reduceLRoP_min_delta or 0
            self.config.reduceLRoP_min_lr = self.config.reduceLRoP_min_lr or 0

            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor = 'val_PR-AUC',
                    mode = 'max',
                    factor = self.config.reduceLRoP_factor,
                    min_delta = self.config.reduceLRoP_min_delta,
                    min_lr = self.config.reduceLRoP_min_lr,
                    patience = self.config.reduceLRoP_patience,
                    verbose = 1,
                ),
            )

        history = self.model.fit(self.train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, validation_data=self.valid_dataset)

        return history.history

    @tf.function
    def _train_step(self, batch, metrics=None):

        def _train_step_per_replica(batch, metrics=None):
            # unpack batch
            features, labels = batch

            with tf.GradientTape() as tape:
                # get model predictions
                logits = self.model(features)
                
                # compute loss
                loss = self.loss(labels, logits) / self.config.batch_size
            
            # apply gradients using optimizer
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            # update metrics
            if metrics:
                for metric in metrics:
                    metric.update_state(labels, logits)
            
            return loss

        # run train step on each replica using distribution strategy
        per_replica_losses = self.strategy.experimental_run_v2(_train_step_per_replica, args=(batch, metrics))
        
        # compute mean loss
        mean_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

        return mean_loss

    @tf.function
    def _valid_step(self, batch, metrics=None):

        def _valid_step_per_replica(batch, metrics=None):
            # unpack batch
            features, labels = batch

            # get model predictions
            logits = self.model(features, training=False)

            # compute loss
            loss = self.loss(labels, logits) / self.config.batch_size

            # update metrics
            if metrics:
                for metric in metrics:
                    metric.update_state(label_batch, logits)
            
            return loss
        
        # run valid step on each replica using distribution strategy
        per_replica_losses = self.strategy.experimental_run_v2(_valid_step_per_replica, args=(batch, metrics))
        
        # compute mean loss
        mean_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

        return mean_loss

    def train(self, epochs, steps_per_epoch=None, restore=None, update_freq=1, analyse_trace=False):
        ''' Creates a compiled instance of the training model and trains it for 'epochs' epochs using a custom training loop.

        Parameters
        ----------
        train_dataset: tf.data.Dataset
            The training dataset.
            
        valid_dataset: tf.data.Dataset
            The validation dataset. If None, validation will be disabled. Tfe callbacks might not work properly.

        frontend: {'waveform', 'log-mel-spectrogram'}
            The frontend to adopt.
            
        strategy: tf.distribute.Strategy
            Strategy for multi-GPU distribution.

        config: argparse.Namespace
            Instance of the config namespace. It is generated when parsing the config.json file.
            
        epochs: int
            Specifies the number of epochs to train for.
        
        restore: str
            Specifies the timestamp of the checkpoint to restore. Should be a timestamp in the 'YYMMDD-hhmm' format.

        update_freq: int
            Specifies the number of batches to wait before writing to logs. Note that writing too frequently can slow down training.
        
        analyse_trace: bool
            Specifies whether to enable profiling.
        '''

        train_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)
        valid_dataset = self.strategy.experimental_distribute_dataset(self.valid_dataset) if self.valid_dataset else None
        
        with self.strategy.scope():
            
            # get previous optimizer iteration count
            opt_init_count = self.optimizer.iteration.numpy() # optimizer *does not* reset on train keyboard interrupt, but only when the class goes out of scope

            # get previous epoch count (in case a checkpoint is restored)
            start = self.checkpoint.save_counter

            # initialize early stop variables
            early_stop = 0
            early_stop_max_metric = 0

            for epoch in tf.range(start, epochs, dtype=tf.int64):

                print()
                print('Epoch {}/{}'.format(epoch+1, epochs))
                
                if analyse_trace and epoch == 0:
                    tf.summary.trace_off()
                    tf.summary.trace_on(graph=False, profiler=True)
                
                # train
                for step, batch in enumerate(train_dataset):
                    loss = self._train_step(batch, metrics=[self.metric_1, self.metric_2])
                    print('{:4d} - Loss {:8.5f} - ROC-AUC {:6.5f} - PR-AUC {:6.5f}'.format(step+1, loss, self.metric_1.result(), self.metric_2.result()), end='\r')

                    # write metrics on tensorboard every update_freq step
                    if tf.equal(self.optimizer.iterations % update_freq, 0):
                        # if lr_callback:
                        #     self.optimizer.learning_rate.assign(lr_callback.get_lr(self.optimizer.iterations))
 
                        with self.train_summary_writer.as_default():
                            tf.summary.scalar(name='iter_loss',
                                            data=loss, 
                                            step=self.optimizer.iterations.numpy())
                            tf.summary.scalar(name='iter_ROC-AUC', 
                                            data=self.metric_1.result(),
                                            step=self.optimizer.iterations.numpy())
                            tf.summary.scalar(name='iter_PR-AUC', 
                                            data=self.metric_2.result(),
                                            step=self.optimizer.iterations.numpy())
                            self.train_summary_writer.flush()

                # print progress summary
                print('Loss {:8.5f} - ROC-AUC {:6.5f} - PR-AUC {:6.5f}'.format(loss, self.metric_1.result(), self.metric_2.result()))
               
                # write metrics on tensorboard at the end of each epoch
                with self.train_summary_writer.as_default():
                    tf.summary.scalar(name='epoch_loss',
                                      data=loss, 
                                      step=epoch+1)
                    tf.summary.scalar(name='epoch_ROC-AUC', 
                                      data=self.metric_1.result(),
                                      step=epoch+1)
                    tf.summary.scalar(name='epoch_PR-AUC', 
                                      data=self.metric_2.result(),
                                      step=epoch+1)
                    self.train_summary_writer.flush()
                
                # reset
                self.metric_1.reset_states()
                self.metric_2.reset_states()

                # write training profile
                if analyse_trace:
                    with self.profiler_summary_writer.as_default():   
                        tf.summary.trace_export(name="trace", 
                                                step=epoch+1, 
                                                profiler_outdir=self.profiler_log_dir_log_dir)

                # save checkpoint
                checkpoint.save(os.path.join(self.log_dir, 'epoch_' + str(epoch+1)))

                if valid_dataset:
                    
                    print()
                    print('Epoch {}/{} - Validation'.format(epoch+1, epochs))
                    
                    # validate
                    for step, batch in enumerate(valid_dataset):
                        loss = self._valid_step(batch, metrics=[self.metric_1, self.metric_2])
                        print('{:4d} - Loss {:8.5f} - ROC-AUC {:6.5f} - PR-AUC {:6.5f}'.format(step+1, loss, self.metric_1.result(), self.metric_2.result()), end='\r')
                    
                    # print progress summary
                    print('Loss {:8.5f} - ROC-AUC {:6.5f} - PR-AUC {:6.5f}'.format(loss, self.metric_1.result(), self.metric_2.result()))
                    print()
                    
                    # write metrics on tensorboard at the end of each epoch
                    with self.valid_summary_writer.as_default():
                        tf.summary.scalar(name='epoch_loss',
                                        data=loss, 
                                        step=epoch+1)
                        tf.summary.scalar(name='epoch_ROC-AUC', 
                                        data=self.metric_1.result(),
                                        step=epoch+1)
                        tf.summary.scalar(name='epoch_PR-AUC', 
                                        data=self.metric_2.result(),
                                        step=epoch+1)
                        self.valid_summary_writer.flush()
                
                    # early stop check
                    if self.config.early_stop_patience is not None:

                        self.config.early_stop_min_delta = self.config.early_stop_min_delta or 0.

                        if self.metric_2.result() > (early_stop_max + self.config.early_stop_min_d):
                            early_stop_max_metric = self.metrics_2.result()
                            early_stop = 0
                        else:
                            early_stop += 1
                            print('No significant improvements on PR-AUC ({:1d}/{:1d})'.format(early_stop, self.config.early_stop_patience))
                            if early_stop == self.config.early_stop_patience:
                                break
                    
                    # reset
                    self.metric_1.reset_states()
                    self.metric_2.reset_states()
    
    def lr_find_plot(self):
        try:
            fig = plt.figure()
            plt.plot(self.lr_find_x, self.lr_find_y)
            plt.yscale('log')
            plt.xscale('log')
        except AttributeError:
            print('you need to run lr_find() first, in order to plot the results with lr_find_plot()')
    
    def lr_find(self, start_lr=1e-07, end_lr=10, num_it=1000, stop_div=True):
        
        learning_rate = start_lr
        
        f = (end_lr/start_lr)**(1/num_it) # the factor to multiply lr by after each iteration
        
        self.lr_find_x = []
        self.lr_find_y = []

        for step, batch in enumerate(self.train_dataset):
            # update learning rate
            learning_rate *= f
            self.optimizer.learning_rate.assign(learning_rate)

            # get loss
            loss = self._train_step(batch)
            print('{:4d} - Learning Rate {:.3e} - Loss {:8.5f}'.format(step+1, learning_rate, loss), end='\r')

            # add loss to output
            self.lr_find_x.append(learning_rate)
            self.lr_find_y.append(loss)

            # stop
            if step >= num_it-1:
                break

            # stop if loss is diverging
            if stop_div:
                if self.lr_find_y[-1] - self.lr_find_y[0] > self.lr_find_y[0]/5:
                    self.lr_find_x.pop()
                    self.lr_find_y.pop()
                    break

class Scheduler:
    def __init__(self, cycle_length, max_lr=0.1, div_factor=10, moms=None):
        self.max_lr = max_lr
        self.min_lr = max_lr / div_factor
        self.delta = self.max_lr - self.min_lr
        self.moms = moms
        self.cycle_length = cycle_length

    def _step_fn(self, step, max_val, min_val, reverse=False): # overridden by subclass
        pass
    
    def get_lr(self, step):
        return self._step_fn(step, self.max_lr, self.min_lr)

    def get_mm(self, step):
        max_mm, min_mm = self.moms
        return self._step_fn(step, max_mm, min_mm, reverse=True)

class CyclicLR(Scheduler):
    def __init__(self, max_lr, cycle_length, div_factor=10, moms=(0.95, 0.85)):
        super().__init__(cycle_length, max_lr, div_factor, moms)

    def _step_fn(self, step, max_val, min_val, reverse=False):
        delta = max_val - min_val

        # get index of current cycle (i.e. how many full cycles have already been completed)
        cycle = math.floor(step/self.cycle_length)

        # get progress ratio within current cycle (i.e. a float between 0 and 1)
        ratio = step/self.cycle_length - cycle

        # get progress ratio within current cycle, but count from the nearest extremum (i.e. a float between 0 and .5)
        effective_ratio = (0.5 - abs(0.5 - ratio)) * 2

        start = min_val if not reverse else max_val

        f = int(reverse) * 2 - 1

        val = start - f * (delta * effective_ratio)

        return val

class CyclicLR_1Cycle(Scheduler):
    def __init__(self, max_lr, cycle_length, div_factor=10, moms=(0.95, 0.85)):
        super().__init__(cycle_length, max_lr, div_factor, moms)

    def _step_fn(self, step, max_val, min_val, reverse=False):
        delta_1 = max_val - min_val
        delta_2 = max_val - math.floor(min_val) if not reverse else max_val - min_val

        ratio = step / self.cycle_length

        assert ratio <= 1

        scaled_ratio_1 = ratio * 5 / 2
        scaled_ratio_2 = ratio * 5 / 3 - 2 / 3

        f = int(reverse) * 2 - 1

        if not reverse:
            if ratio < 0.4:
                val = delta_1 * scaled_ratio_1 * (-f) + min_val
            else:
                val = delta_2 / 2 * (1 + math.cos(math.pi * scaled_ratio_2)) * (-f)
        else:
            if ratio < 0.4:
                val = delta_1 * scaled_ratio_1 * (-f) + max_val
            else:
                val = delta_2 / 2 * (1 + math.cos(math.pi * scaled_ratio_2)) * (-f) + max_val

        return val

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('frontend', choices=['waveform', 'log-mel-spectrogram'])
    parser.add_argument('--tfrecords-dir', help='directory to read the .tfrecord files from', required=True)
    parser.add_argument('--config', help='path to config.json (default to path on Boden)', default='/srv/data/urop/config.json')
    parser.add_argument('--lastfm', help='path to (clean) lastfm database (default to path on Boden)', default='/srv/data/urop/clean_lastfm.db')
    parser.add_argument('--multi-db', help='specify the number of different tags features in the .tfrecord files', type=int, default=1)
    parser.add_argument('--multi-db-default', help='specify the index of the default tags database, when there are more than one tags features in the .tfrecord files', type=int)
    parser.add_argument('--epochs', help='specify the number of epochs to train for', type=int, default=1)
    parser.add_argument('--steps-per-epoch', help='specify the number of steps to perform at each epoch (if unspecified, go through the whole dataset)', type=int)
    parser.add_argument('--no-shuffle', action='store_true', help='force no shuffle, override config setting')
    parser.add_argument('--restore', help='load a previously saved model with the time in the format ddmmyy-hhmm, e.g. if the folder which the model is saved is custom_log-mel-spect_160919-0539, resume should take the argument 160919-0539')
    parser.add_argument('--update-freq', help='specify the frequency (in steps) to record metrics and losses', type=int, default=10)
    parser.add_argument('--cuda', help='set cuda visible devices', type=int, nargs='+')
    parser.add_argument('--built-in', action='store_true', help='train using the built-in model.fit training loop')
    parser.add_argument('-v', '--verbose', choices=['0', '1', '2', '3'], help='verbose mode', default='1')

    args = parser.parse_args()

    # specify number of visible gpu's
    if args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in args.cuda])

    # specify verbose mode
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.verbose

    # parse config
    config = parse_config_json(args.config, args.lastfm)

    # override config setting
    if args.no_shuffle:
        config.shuffle = False

    # generate train_dataset and valid_dataset (valid_dataset will be None if config.split is None)
    train_dataset, valid_dataset = generate_datasets_from_dir(args.tfrecords_dir, args.frontend, split = config.split, which_split=(True, True, ) + (False, ) * (len(config.split)-2),
                                                              sample_rate = config.sr, batch_size = config.batch_size, 
                                                              block_length = config.interleave_block_length, cycle_length = config.interleave_cycle_length,
                                                              shuffle = config.shuffle, shuffle_buffer_size = config.shuffle_buffer_size, 
                                                              window_length = config.window_length, window_random = config.window_random, 
                                                              hop_length = config.melspect_x_hop_length, num_mel_bands = config.melspect_y, tag_shape = config.tag_shape, with_tags = config.tags,
                                                              num_tags_db = args.multi_db, default_tags_db = args.multi_db_default,
										                      as_tuple = True)
    
    strategy = tf.distribute.MirroredStrategy()

    # instantiate learner
    orpheus = Learner(frontend=args.frontend, 
                      train_dataset=train_dataset, valid_dataset=valid_dataset, 
                      strategy=strategy, config=config, 
                      restore=args.restore, custom_loop=(not args.built_in))

    # train
    if not args.built_in:
        orpheus.train(epochs=args.epochs,
                      steps_per_epoch=args.steps_per_epoch,
                      restore=args.restore,
                      update_freq=args.update_freq)
    else:
        orpheus.train_with_fit(epochs=args.epochs, 
                        steps_per_epoch=args.steps_per_epoch, 
                        restore=args.restore, 
                        update_freq=args.update_freq)