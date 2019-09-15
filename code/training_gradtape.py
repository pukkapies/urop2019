''' Contains tools to train our model using the mirrored distribution strategy and a custom training loop.


Notes
-----
This module contains the actual training function used to train a model
using the custom GradientTape API. It combines the data input pipeline 
defined in projectame_input.py (which uses the .tfrecord files previously
generated by audio_processing.py), the CNN architecture proposed 
by Pon et. al (2018) defined in projectname.py, and a custom training loop 
with mirrored strategy integrated for multiple-GPU training.

The training function tries to optimise BinaryCrossentropy for each tag
prediction (batch loss is summed over the batch size, instead of being averaged), and 
displays the area under the ROC and PR curves as metrics. The optimizer can
be fully specified in the config.json file.

The logs are saved automatically subdirectories named after the timestamp and can be accessed
using TensorBoard. The checkpoints are saved automatically in subdirectories named after
the frontend being adopted and timestamp. The config.json file is copied automatically
in the latter directory for future reference. By recovering a checkpoint using resume_time, 
training will resume from the latest completed epoch. 

If early stopping is enabled, a .npy file will be generated to
store the early stopping progress in case the script is stopped and resumed
later.

IMPORTANT: if trying to profile a batch in TensorBoard, make sure the environment
variable LD_LIBRARY_PATH is specified.
(e.g. 'export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:
                               /usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64"')


Functions
---------
- train
    Creates a compiled instance of the training model and trains it. 
'''

import argparse
import datetime
import gc
import json
import os
import time
import shutil

import numpy as np
import tensorflow as tf

import projectname
import projectname_input

from lastfm import LastFm

def parse_config(config_path, lastfm_path):

    # load tags database
    lastfm = LastFm(os.path.expanduser(lastfm_path))

    if not os.path.isfile(os.path.expanduser(config_path)):
        path = os.path.join(os.path.abspath(os.path.expanduser(config_path)), 'config.json')
    else:
        path = os.path.expanduser(config_path)

    # load json
    with open(path, 'r') as f:
        config_d = json.loads(f.read())

    
    n_tags = config_d['tfrecords']['n_tags']
    # read top tags from popularity dataframe
    top = config_d['tags']['top']
    if (top is not None) and (top !=n_tags):
        top_tags = lastfm.popularity()['tag'][:top].tolist()
        tags = set(top_tags)
    else:
        tags=None

    # find tags to use
    if tags is not None:
        if config_d['tags']['with']:
            tags.union(config_d['tags']['with'])
        
        if config_d['tags']['without']:
            tags.discard(config_d['tags']['without'])
    else:
        raise ValueError("parameter 'with' is inconsistent to parameter 'top'")

    # create config namespace (to be accessed more easily than a dictionary)
    config = argparse.Namespace()
    config.batch = config_d['config']['batch_size']
    config.cycle_len = config_d['config']['cycle_length']
    config.early_stop_min_d = config_d['config']['early_stop_min_delta']
    config.early_stop_patience = config_d['config']['early_stop_patience']
    config.n_dense_units = config_d['model']['n_dense_units']
    config.n_filters = config_d['model']['n_filters']
    config.n_mels = config_d['tfrecords']['n_mels']
    config.n_output_neurons = len(tags) if tags is not None else n_tags
    config.path = config_path
    config.plateau_min_d = config_d['config']['reduce_lr_plateau_min_delta']
    config.plateau_patience = config_d['config']['reduce_lr_plateau_patience']
    config.shuffle = config_d['config']['shuffle']
    config.shuffle_buffer = config_d['config']['shuffle_buffer_size']
    config.split = config_d['config']['split']
    config.sr = config_d['tfrecords']['sample_rate']
    config.tags = lastfm.tag_to_tag_num(list(tags)) if tags is not None else None
    config.tags_to_merge = lastfm.tag_to_tag_num(config_d['tags']['merge']) if config_d['tags']['merge'] is not None else None
    config.tot_tags = config_d['tfrecords']['n_tags']
    config.window_len = config_d['config']['window_length']
    config.window_random = config_d['config']['window_extract_randomly']
    config.log_dir = config_d['config']['log_dir']
    config.checkpoint_dir = config_d['config']['checkpoint_dir']

    # create config namespace for the optimizer (will be used by get_optimizer() in order to allow max flexibility)
    config_optim = argparse.Namespace()
    config_optim.class_name = config_d['optimizer'].pop('name')
    config_optim.max_learning_rate = config_d['optimizer'].pop('max_learning_rate')
    config_optim.cycle_stepsize = config_d['optimizer'].pop('cycle_stepsize')
    config_optim.config = config_d['optimizer']
    
    return config, config_optim
            
def train(train_dataset, valid_dataset, frontend, strategy, config, config_optim, epochs, resume_time=None, update_freq=1, analyse_trace=False):
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
    
    config_optim: argparse.Namespace
        Instance of the config_optim namespace. The optimizer will be fully specified by this parameter. It is generated when parsing the config.json file.
        
    epochs: int
        Specifies the number of epochs to train for.
    
    resume_time: str
        Specifies the timestamp of the checkpoint to restore. Should be a timestamp in the 'YYMMDD-hhmm' format.

    update_freq: int
        Specifies the number of batches to wait before writing to logs. Note that writing too frequently can slow down training.
    
    analyse_trace: bool
        Specifies whether to enable profiling.
    '''

    log_dir = os.path.join(os.path.expanduser(config.log_dir), datetime.datetime.now().strftime("%y%m%d-%H%M%S")) # to save training metrics (to access using tensorboard)
    checkpoint_dir = os.path.join(os.path.join(os.path.expanduser(config.checkpoint_dir), frontend + '_' + datetime.datetime.now().strftime("%y%m%d-%H%M%S"))) # to save model checkpoints
    
    with strategy.scope():
        
        num_replica = strategy.num_replicas_in_sync
        tf.print('num_replica:', num_replica)
        
        # build model
        model = projectname.build_model(frontend, num_output_neurons=config.n_output_neurons, num_units=config.n_dense_units, num_filts=config.n_filters, y_input=config.n_mels)
        
        # initialise loss, optimizer and metrics

        def get_learning_rate(step, cycle_stepsize, max_lr):
            # it is recommended that min_lr is 1/3 or 1/4th of the maximum lr. see:  
            min_lr = max_lr/4
            current_cycle = tf.floor(step/(2*cycle_stepsize))
            ratio = step/cycle_stepsize-current_cycle*2
            lr = min_lr + (max_lr - min_lr)*tf.cast(tf.abs(tf.abs(ratio-1)-1), dtype=tf.float32)
            return lr

        if config_optim.max_learning_rate and config_optim.cycle_stepsize:
            config_optim.config['learning_rate'] = tf.Variable(get_learning_rate(0, config_optim.cycle_stepsize, config_optim.max_learning_rate))

        optimizer = tf.keras.optimizers.get({"class_name": config_optim.class_name, "config": config_optim.config})
        train_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        train_mean_loss = tf.keras.metrics.Mean(name='train_mean_loss', dtype=tf.float32)
        train_metrics_1 = tf.keras.metrics.AUC(curve='ROC', name='train_AUC-ROC', dtype=tf.float32)
        train_metrics_2 = tf.keras.metrics.AUC(curve='PR', name='train_AUC-PR', dtype=tf.float32)

        # setting up checkpoint
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        prev_epoch = -1
        
        # resume
        if resume_time is None:
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            shutil.copy(config.path, checkpoint_dir) # copy config file in the same folder where the models will be saved
        else:
            log_dir = os.path.join(os.path.expanduser(config.log_dir), resume_time) # keep saving logs in the 'old' folder
            checkpoint_dir = os.path.join(os.path.expanduser(config.checkpoint_dir), frontend + '_' + resume_time) # keep saving checkpoints in the 'old' folder
            
            # try to load checkpoint
            chkp = tf.train.latest_checkpoint(checkpoint_dir)
            if chkp:
                tf.print("Checkpoint file {} found. Restoring...".format(chkp))
                checkpoint.restore(chkp)
                tf.print("Checkpoint restored.")
                prev_epoch = int(chkp.split('-')[-1])-1  #last completed epoch number (from 0)
            else:
                tf.print("Checkpoint file not found!")
                return
        
        tf.summary.trace_off() # in case of previous keyboard interrupt
        
        # setting up summary writers
        train_log_dir = os.path.join(log_dir, 'train/')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        
        if valid_dataset:
            val_log_dir = os.path.join(log_dir, 'validation/')
            val_summary_writer = tf.summary.create_file_writer(val_log_dir)
            val_metrics_1 = tf.keras.metrics.AUC(curve = 'ROC', name='val_AUC-ROC', dtype=tf.float32)
            val_metrics_2 = tf.keras.metrics.AUC(curve = 'PR', name='val_AUC-PR', dtype=tf.float32)
            val_loss = tf.keras.metrics.Mean(name='val_loss', dtype=tf.float32)
        
        if analyse_trace: # make sure the variable LD_LIBRARY_PATH is properly set up
            print('TIPS: To ensure the profiler works correctly, make sure the LD_LIBRARY_PATH is set correctly. \
                  For Boden, set--- export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64" before Python is initialised.')
            prof_log_dir = os.path.join(log_dir, 'profile/')
            prof_summary_writer = tf.summary.create_file_writer(prof_log_dir)

        
        # rescale loss
        def compute_loss(labels, predictions):
            per_example_loss = train_loss(labels, predictions)
            return per_example_loss/config.batch
        
        def train_step(entry):
            audio_batch, label_batch = entry['audio'], entry['tags']
            with tf.GradientTape() as tape:
                logits = model(audio_batch)
                loss = compute_loss(label_batch, logits)
            variables = model.trainable_variables
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            train_metrics_1.update_state(label_batch, logits)
            train_metrics_2.update_state(label_batch, logits)
            train_mean_loss.update_state(loss)
            return loss

        def valid_step(entry):
            audio_batch, label_batch = entry['audio'], entry['tags']
            logits = model(audio_batch, training=False)
            loss = compute_loss(label_batch, logits)

            val_metrics_1.update_state(label_batch, logits)
            val_metrics_2.update_state(label_batch, logits)
            val_loss.update_state(loss)
            return loss
            
        @tf.function 
        def distributed_train_body(entry, epoch, num_replica):
            num_batches = 0 
            for entry in train_dataset:
                tf.print('Learning rate ', optimizer.learning_rate)
                strategy.experimental_run_v2(train_step, args=(entry, ))
                optimizer.learning_rate.assign(get_learning_rate(optimizer.iterations, config_optim.cycle_stepsize, config_optim.max_learning_rate))
                num_batches += 1
                # print metrics after each iteration
                if tf.equal(num_batches % update_freq, 0):
                    tf.print('Epoch',  epoch,'; Step', num_batches, '; loss', tf.multiply(train_mean_loss.result(), num_replica), '; ROC_AUC', train_metrics_1.result(), ';PR_AUC', train_metrics_2.result())

                    with train_summary_writer.as_default():
                        tf.summary.scalar('ROC_AUC_itr', train_metrics_1.result(), step=optimizer.iterations)
                        tf.summary.scalar('PR_AUC_itr', train_metrics_2.result(), step=optimizer.iterations)
                        tf.summary.scalar('Loss_itr', tf.multiply(train_mean_loss.result(), num_replica), step=optimizer.iterations)
                        train_summary_writer.flush()
                gc.collect()

        @tf.function
        def distributed_val_body(entry):
            for entry in valid_dataset:
                strategy.experimental_run_v2(valid_step, args=(entry, ))
                gc.collect()

        max_metric = -200 # for early stopping

        
        # loop
        for epoch in tf.range(prev_epoch+1, epochs, dtype=tf.int64):
            start_time = time.time()
            tf.print()
            tf.print()
            tf.print('Epoch {}/{}'.format(epoch, epochs-1))
            
            if analyse_trace and tf.equal(epoch, 1):
                tf.summary.trace_off()
                tf.summary.trace_on(graph=False, profiler=True)
            
            distributed_train_body(train_dataset, epoch, num_replica)
            gc.collect()
            
            # write metrics on tensorboard after each epoch
            with train_summary_writer.as_default():
                tf.summary.scalar('ROC_AUC_epoch', train_metrics_1.result(), step=epoch)
                tf.summary.scalar('PR_AUC_epoch', train_metrics_2.result(), step=epoch)
                tf.summary.scalar('mean_loss_epoch', tf.multiply(train_mean_loss.result(), num_replica), step=epoch)
                train_summary_writer.flush()
                
            # print progress
            tf.print('Epoch', epoch,  ': loss', tf.multiply(train_mean_loss.result(), num_replica), '; ROC_AUC', train_metrics_1.result(), '; PR_AUC', train_metrics_2.result())
            
            train_metrics_1.reset_states()
            train_metrics_2.reset_states()
            train_mean_loss.reset_states()

            # export profiling and write validation metrics on tensorboard
            if analyse_trace:
                with prof_summary_writer.as_default():   
                    tf.summary.trace_export(name="trace", 
                                            step=epoch, 
                                            profiler_outdir=os.path.normpath(prof_log_dir)) 

            if valid_dataset:
                distributed_val_body(valid_dataset)
                gc.collect()
                with val_summary_writer.as_default():
                    tf.summary.scalar('ROC_AUC_epoch', val_metrics_1.result(), step=epoch)
                    tf.summary.scalar('PR_AUC_epoch', val_metrics_2.result(), step=epoch)
                    tf.summary.scalar('mean_loss_epoch', tf.multiply(val_loss.result(), num_replica), step=epoch)
                    val_summary_writer.flush()

                tf.print('Val- Epoch', epoch, ': loss', tf.multiply(val_loss.result(), num_replica), ';ROC_AUC', val_metrics_1.result(), '; PR_AUC', val_metrics_2.result())
                
                # early stopping
                if (config.early_stop_min_d) or (config.early_stop_patience):
                    
                    if not config.early_stop_min_d:
                        config.early_stop_min_d = 0.
                   
                    if not config.early_stop_patience:
                        config.early_stop_patience = 1
                    
                    if os.path.isfile(os.path.join(checkpoint_dir, 'early_stopping.npy')):
                        cumerror = int(np.load(os.path.join(checkpoint_dir, 'early_stopping.npy')))
                    
                    if val_metrics_2.result() > (max_metric + config.early_stop_min_d):
                        max_metric = val_metrics_2.result()
                        cumerror = 0
                        np.save(os.path.join(checkpoint_dir, 'early_stopping.npy'), cumerror)
                    else:
                        cumerror += 1
                        tf.print('Epoch {}/{}: no significant improvements ({}/{})'.format(epoch, epochs-1, cumerror, config.early_stop_patience))
                        np.save(os.path.join(checkpoint_dir, 'early_stopping.npy'), cumerror)
                        if cumerror == config.early_stop_patience:
                            tf.print('Epoch {}: stopping')
                            break
                
                # reset validation metrics after each epoch
                val_metrics_1.reset_states()
                val_metrics_2.reset_states()
                val_loss.reset_states()
                    
            elif (config.early_stop_min_d) or (config.early_stop_patience):
                raise RuntimeError('EarlyStopping requires a validation dataset')

            checkpoint_path = os.path.join(checkpoint_dir, 'epoch'+str(epoch.numpy()))
            saved_path = checkpoint.save(checkpoint_path)
            tf.print('Saving model as TF checkpoint: {}'.format(saved_path))

            # report time
            time_taken = time.time()-start_time
            tf.print('Epoch {}: {} s'.format(epoch, time_taken))
            
            tf.keras.backend.clear_session()
            gc.collect()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("frontend", choices=["waveform", "log-mel-spectrogram"])
    parser.add_argument("--root-dir", dest="tfrecords_dir", help="directory to read .tfrecord files from (default to path on Boden)")
    parser.add_argument("--config-path", help="path to config.json (default to path on Boden)", default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json'))
    parser.add_argument("--lastfm-path", help="path to (clean) lastfm database (default to path on Boden)", default="/srv/data/urop/clean_lastfm.db")
    parser.add_argument("--epochs", help="specify the number of epochs to train on", type=int, required=True)
    parser.add_argument("--steps-per-epoch", help="specify the number of steps to perform for each epoch (if unspecified, go through the whole dataset)", type=int)
    parser.add_argument("--no-shuffle", action="store_true", help="override shuffle setting")
    parser.add_argument("--resume-time", help="load a previously saved model")
    parser.add_argument("--update-freq", help="specify the frequency (in steps) to record metrics and losses", type=int, default=10)
    parser.add_argument("--cuda", help="set cuda visible devices", type=int, nargs="+")
    parser.add_argument("-v", "--verbose", choices=['0', '1', '2', '3'], help="verbose mode", default='0')

    args = parser.parse_args()

    # specify number of visible gpu's
    if args.cuda:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in args.cuda])

    # specify verbose mode
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.verbose

    # parse json
    config, config_optim = parse_config(args.config_path, args.lastfm_path)

    # if root_dir is not specified, use default path on our server
    if not args.tfrecords_dir:
        if config.sr != 16000:
            s = '-' + str(config.sr // 1000) + 'kHz'
        else:
            s = ''
        args.tfrecords_dir = os.path.normpath("/srv/data/urop/tfrecords-" + args.frontend + s)

    # override shuffle setting
    if args.no_shuffle:
        config.shuffle = False

    # create training and validation dataset
    assert config.split
    assert len(config.split) >= 2
    assert len(config.split) <= 3
    train_dataset, valid_dataset = projectname_input.generate_datasets_from_dir(args.tfrecords_dir, args.frontend, split=config.split, which_split=(True, True, ) + (False, ) * (len(config.split)-2),
                                                                                sample_rate=config.sr, batch_size=config.batch, 
                                                                                cycle_length=config.cycle_len, 
                                                                                shuffle=config.shuffle, shuffle_buffer_size=config.shuffle_buffer, 
                                                                                num_tags=config.tot_tags, window_length=config.window_len, window_random=config.window_random, 
                                                                                with_tags=config.tags, merge_tags=config.tags_to_merge,
										                                        as_tuple=False)
    
    if args.steps_per_epoch:
        train_dataset = train_dataset.take(args.steps_per_epoch)
        valid_dataset = valid_dataset.take(args.steps_per_epoch)

    # set up training strategy
    strategy = tf.distribute.MirroredStrategy()
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    valid_dataset = strategy.experimental_distribute_dataset(valid_dataset)

    # train
    train(train_dataset, valid_dataset, frontend=args.frontend, strategy=strategy,
          config=config, config_optim=config_optim,
          epochs=args.epochs, resume_time=args.resume_time, 
          update_freq=args.update_freq)

