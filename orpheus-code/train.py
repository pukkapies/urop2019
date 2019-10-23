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
import os
import shutil

import tensorflow as tf

from data_input import generate_datasets_from_dir
from orpheus_model import build_model
from orpheus_model import parse_config_json

class Learner():
    def __init__(frontend, train_dataset, valid_dataset, strategy, config, restore=None, custom_loop=True):
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
            self.model = build_model(frontend, num_output_neurons=config.num_output_neurons, num_dense_units=config.num_dense_units, y_input=config.melspect_y)

            self.optimizer = tf.keras.optimizers.get({"class_name": config.optimizer_name, "config": config.optimizer})

            self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM)

            self.train_metric_1 = tf.keras.metrics.AUC(curve='ROC',
                                                       name='train_ROC-AUC', 
                                                       dtype=tf.float32)
            self.train_metric_2 = tf.keras.metrics.AUC(curve='PR',
                                                       name='train_PR-AUC',
                                                       dtype=tf.float32)

            if custom_loop:
                # initialize (and restore) checkpoint
                self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
            
                if restore:
                    file = tf.train.latest_checkpoint(self.log_dir)
                    if file:
                        self.checkpoint.restore(file)
                        self.checkpoint_epoch = int(file.split('-')[-1]) # restart training from last saved epoch
                    else:
                        raise FileNotFoundError
                
                # initialize validation metrics (automatically done in built-in loop)
                self.valid_metric_1 = tf.keras.metrics.AUC(curve='ROC',
                                                        name='train_ROC-AUC', 
                                                        dtype=tf.float32)
                self.valid_metric_2 = tf.keras.metrics.AUC(curve='PR',
                                                        name='train_PR-AUC',
                                                        dtype=tf.float32)
                
                # initialize tensorboard summary writers
                self.train_log_dir = os.path.join(self.log_dir, 'train/')
                self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
                self.valid_log_dir = os.path.join(self.log_dir, 'validation/')
                self.valid_summary_writer = tf.summary.create_file_writer(self.valid_log_dir)
                self.profiler_log_dir = os.path.join(self.log_dir, 'profile/')
                self.profiler_summary_writer = tf.summary.create_file_writer(self.profiler_log_dir)

    def train_with_fit(self, epochs, steps_per_epoch=None, restore=None, update_freq=1, profile_batch=0):
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
                self.model.load_weights(os.path.join(os.path.expanduser(self.config.checkpoint_dir), self.frontend + '_' + restore))

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
                profile_batch = profile_batch, # make sure the env variable LD_LIBRARY_PATH is properly set up
            ),

            tf.keras.callbacks.TerminateOnNaN(),
        ]

        if self.config.early_stop_patience is not None:

            self.config.early_stop_min_delta = self.config.early_stop_min_delta or 0

            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor = 'val_PR-AUC',
                    mode = 'max',
                    min_delta = config.early_stop_min_delta,
                    restore_best_weights = True,
                    patience = config.early_stop_patience,
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
    def _train_step(self, batch, strategy, metrics=None):

        def _train_step_per_replica(batch, metrics=None):
            # unpack batch
            features, labels = batch

            with tf.GradientTape() as tape:
                # get model predictions
                logits = self.model(features)
                
                # compute loss
                loss = self.train_loss(labels, logits) / self.config.batch_size
            
            # apply gradients using optimizer
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            # update metrics
            if metrics:
                for metric in metrics:
                    metric.update_state(labels, logits)
            
            return loss

        # run train step on each replica using distribution strategy
        per_replica_losses = strategy.experimental_run_v2(_train_step_per_replica, args=(batch, metrics))
        
        # compute mean loss
        mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

        return mean_loss

    @tf.function
    def _valid_step(self, batch, strategy, metrics=None):

        def _valid_step_per_replica(batch, metrics=None):
            # unpack batch
            features, labels = batch

            # get model predictions
            logits = self.model(features, training=False)

            # compute loss
            loss = self.train_loss(labels, logits) / self.config.batch_size

            # update metrics
            if metrics:
                for metric in metrics:
                    metric.update_state(label_batch, logits)
            
            return loss
        
        # run valid step on each replica using distribution strategy
        per_replica_losses = strategy.experimental_run_v2(_valid_step_per_replica, args=(batch, metrics))
        
        # compute mean loss
        mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

        return mean_loss

    # def get_cyclic_learning_rate(step, iterations, max_lr):
    #     # it is recommended that min_lr is 1/3 or 1/4th of the maximum lr. see:  
    #     min_lr = max_lr/4
    #     # cycle stepsize twice the iterations in an epoch is recommended
    #     cycle_stepsize = iterations*2

    #     current_cycle = tf.floor(step/(2*cycle_stepsize))
    #     ratio = step/cycle_stepsize-current_cycle*2
    #     lr = min_lr + (max_lr - min_lr)*tf.cast(tf.abs(tf.abs(ratio-1)-1), dtype=tf.float32)
    #     return lr

    # def get_range_test_learning_rate(step, lr_range, iterations):
    #     lr = lr_range[0]*tf.cast((lr_range[1]/lr_range[0])**(step/iterations), dtype=tf.float32)
    #     return lr

    # if lr_range:
    #     config_optim.config['learning_rate'] = tf.Variable(get_range_test_learning_rate(0, lr_range, config.iterations))
    # elif config_optim.max_learning_rate:
    #     config_optim.config['learning_rate'] = tf.Variable(get_cyclic_learning_rate(0, config.iterations, config_optim.max_learning_rate))

    def train(self, epochs, steps_per_epoch=None, restore=None, update_freq=1, lr_range=None, analyse_trace=False):
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
        
        with self.strategy.scope():
                
            # @tf.function 
            # def distributed_train_body(entry, epoch, num_replicas):
            #     for entry in train_dataset:
            #         per_replica_losses = strategy.experimental_run_v2(train_step, args=(entry, ))

            #         if lr_range:
            #             optimizer.learning_rate.assign(get_range_test_learning_rate(optimizer.iterations, lr_range, config.iterations))
            #         elif config_optim.max_learning_rate:
            #             optimizer.learning_rate.assign(get_cyclic_learning_rate(optimizer.iterations, config.iterations, config_optim.max_learning_rate))

            #         # print metrics after each iteration
            #         if tf.equal(optimizer.iterations % update_freq, 0):
            #             tf.print('Epoch',  epoch,'; Step', optimizer.iterations, '; loss', tf.multiply(train_mean_loss.result(), num_replicas), 
            #                     '; ROC_AUC', train_metrics_1.result(), ';PR_AUC', train_metrics_2.result(), '; learning rate', optimizer.learning_rate)

            #             with train_summary_writer.as_default():
            #                 tf.summary.scalar('ROC_AUC_itr', train_metrics_1.result(), step=optimizer.iterations)
            #                 tf.summary.scalar('PR_AUC_itr', train_metrics_2.result(), step=optimizer.iterations)
            #                 if lr_range:
            #                     tf.summary.scalar('Learning rate', optimizer.learning_rate, step=optimizer.iterations)
            #                     tf.summary.scalar('Loss_itr', strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None), step=optimizer.iterations)
            #                 else:
            #                     tf.summary.scalar('Loss_itr', tf.multiply(train_mean_loss.result(), num_replicas), step=optimizer.iterations)

            #                 train_summary_writer.flush()

            # max_metric = -200 # for early stopping

            tf.summary.trace_off() # in case of previous keyboard interrupt

            epoch = 0

            try:
                epoch += self.checkpoint_epoch # if model has been restored from checkpoint

            for _ in range(epochs):
                epoch += 1
                start_time = time.time()
                print('Epoch {}/{}'.format(epoch, epochs))
                
                if analyse_trace and epoch == 1:
                    tf.summary.trace_off()
                    tf.summary.trace_on(graph=False, profiler=True)
                
                for i, batch in enumerate(train_dataset):
                    self._train_step(batch, strategy=self.strategy, metrics=[self.train_metric_1, self.train_metric_2])
                
                # write metrics on tensorboard after each epoch
                with train_summary_writer.as_default():
                    tf.summary.scalar('ROC_AUC_epoch', train_metrics_1.result(), step=epoch)
                    tf.summary.scalar('PR_AUC_epoch', train_metrics_2.result(), step=epoch)
                    tf.summary.scalar('mean_loss_epoch', tf.multiply(train_mean_loss.result(), strategy.num_replicas_in_sync), step=epoch)
                    train_summary_writer.flush()
                    
                # print progress
                tf.print('Epoch', epoch,  ': loss', tf.multiply(train_mean_loss.result(), strategy.num_replicas_in_sync), '; ROC_AUC', train_metrics_1.result(), '; PR_AUC', train_metrics_2.result())
                
                train_metrics_1.reset_states()
                train_metrics_2.reset_states()
                train_mean_loss.reset_states()

                # write training profile
                if analyse_trace:
                    with prof_summary_writer.as_default():   
                        tf.summary.trace_export(name="trace", 
                                                step=epoch, 
                                                profiler_outdir=os.path.normpath(prof_log_dir)) 

                if valid_dataset:
                    distributed_val_body(valid_dataset)
                    with val_summary_writer.as_default():
                        tf.summary.scalar('ROC_AUC_epoch', val_metrics_1.result(), step=epoch)
                        tf.summary.scalar('PR_AUC_epoch', val_metrics_2.result(), step=epoch)
                        tf.summary.scalar('mean_loss_epoch', tf.multiply(val_loss.result(), strategy.num_replicas_in_sync), step=epoch)
                        val_summary_writer.flush()

                    tf.print('Val- Epoch', epoch, ': loss', tf.multiply(val_loss.result(), strategy.num_replicas_in_sync), ';ROC_AUC', val_metrics_1.result(), '; PR_AUC', val_metrics_2.result())
                    
                    # early stopping callback
                    if config.early_stop_patience is not None:

                        # if some parameters have not been provided, use default
                        config.early_stop_min_delta = config.early_stop_min_delta or 0.
                        
                        if os.path.isfile(os.path.join(checkpoint_dir, 'early_stopping.npy')):
                            cumerror = int(np.load(os.path.join(checkpoint_dir, 'early_stopping.npy')))

                        if tf.less(config_optim.min_lr_plateau, optimizer.learning_rate):
                            if config_optim.lr_plateau_mult:
                                optimizer.learning_rate.assign(tf.multiply(optimizer.learning_rate, config_optim.lr_plateau_mult))
                            else:
                                optimizer.learning_rate.assign(tf.multiply(optimizer.learning_rate, ))

                        elif val_metrics_2.result() > (max_metric + config.early_stop_min_d):
                            max_metric = val_metrics_2.result()
                            cumerror = 0
                            np.save(os.path.join(log_dir, 'early_stopping.npy'), cumerror)
                        else:
                            cumerror += 1
                            tf.print('Epoch {}/{}: no significant improvements ({}/{})'.format(epoch, epochs-1, cumerror, config.early_stop_patience))
                            np.save(os.path.join(log_dir, 'early_stopping.npy'), cumerror)
                            if cumerror == config.early_stop_patience:
                                tf.print('Epoch {}: stopping')
                                break
                    
                    # reset validation metrics after each epoch
                    val_metrics_1.reset_states()
                    val_metrics_2.reset_states()
                    val_loss.reset_states()
                        
                elif config.early_stop_patience is not None:
                    raise RuntimeError('EarlyStopping requires a validation dataset')

                checkpoint_path = os.path.join(log_dir, 'epoch'+str(epoch.numpy()))
                saved_path = checkpoint.save(checkpoint_path)
                tf.print('Saving model as TF checkpoint: {}'.format(saved_path))

                # report time
                time_taken = time.time()-start_time
                tf.print('Epoch {}: {} s'.format(epoch, time_taken))

        return

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
    parser.add_argument('--resume', help='load a previously saved model with the time in the format ddmmyy-hhmm, e.g. if the folder which the model is saved is custom_log-mel-spect_160919-0539, resume should take the argument 160919-0539')
    parser.add_argument('--update-freq', help='specify the frequency (in steps) to record metrics and losses', type=int, default=10)
    parser.add_argument('--cuda', help='set cuda visible devices', type=int, nargs='+')
    parser.add_argument('--built-in', action='store_true', help='train using the built-in model.fit training loop')
    parser.add_argument('-v', '--verbose', choices=['0', '1', '2', '3'], help='verbose mode', default='0')

    args = parser.parse_args()

    # specify number of visible gpu's
    if args.cuda:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' # see issue #152
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

    # set up training strategy
    strategy = tf.distribute.MirroredStrategy()

    if not args.built_in:
        # datasets need to be manually 'distributed'
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        if valid_dataset is not None:
            valid_dataset = strategy.experimental_distribute_dataset(valid_dataset)
        
        # train model using custom training loop (default choice)
        train(train_dataset, valid_dataset, frontend=args.frontend,
                strategy=strategy, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, 
                config=config,
                update_freq=args.update_freq, restore=args.resume)
    else:
        # train model
        train_with_fit(train_dataset, valid_dataset, frontend=args.frontend,
                strategy=strategy, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, 
                config=config,
                update_freq=args.update_freq, restore=args.resume)
