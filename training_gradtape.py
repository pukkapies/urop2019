'''
Notes
-----
The module contains the ultimate function, main() for performing model 
training. It combines the data input pipeline in projectName_input.py (which 
uses the pre-generated tfrecords file by audio_processing.py), the
network proposed by Pon et. al (2018) in projectname.py, and a customised 
training loop (with validation) with Mirrored Strategy integrated for multiple-
GPU training.

The customised training loop uses the Adam optimizer to minimise losses 
computed by BinaryCrossentropy. The PR AUC and ROC AUC are used as metrics to 
monitor the training progress. Tensorboard is automatically logging the metrics
per 10 batches, and can return profiling information if analyse_trace is set 
True. Finally, a Checkpoint is created and saved in the designated directory
at the end of each epoch. By recovering CheckPoint using checkpoint, the 
training will resume from the latest completed epoch. 

Early stopping is enabled if specified, and a npy file will be generated to
store the early stopping progress in case the script is stopped and resumed
later.

IMPORTANT: The codes are written in tensorflow 2.0.0-beta version.
IMPORTANT: Run export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64" before loading TensorFlow

Functions
---------
- train
    Compile model with optimisers, loss, and train with customed training loops 
    and validation loops.
    
- main
    Combine data input pipeline, networks, train and validation loops to 
    perform model training.
'''

import argparse
import datetime
import gc
import json
import os
import time
import shutil
import sys

import numpy as np
import tensorflow as tf

import projectname
import projectname_input
from modules.query_lastfm import LastFm

def _required_length(nmin, nmax):
        class RequiredLength(argparse.Action):
            def __call__(self, parser, args, values, option_string=None):
                if not nmin<=len(values)<=nmax:
                    msg='argument "{f}" requires between {nmin} and {nmax} arguments'.format(f=self.dest, nmin=nmin, nmax=nmax)
                    raise argparse.ArgumentTypeError(msg)
                setattr(args, self.dest, values)
        return RequiredLength

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

    # read top tags from popularity dataframe
    top = int(config_d['tags']['top'])
    top_tags = lastfm.popularity()['tag'][:top].tolist()
    tags = set(top_tags)

    # find tags to use
    if config_d['tags']['with']:
        tags.union(config_d['tags']['with'])
    if config_d['tags']['without']:
        tags.discard(config_d['tags']['without'])

    # create config namespace (to be accessed more easily than a dictionary)
    config = argparse.Namespace()
    config.batch = config_d['config']['batch_size']
    config.cycle_len = config_d['config']['cycle_length']
    config.early_stop_min_d = config_d['config']['early_stop_min_delta']
    config.early_stop_patience = config_d['config']['early_stop_patience']
    config.n_dense_units = config_d['model']['n_dense_units']
    config.n_filters = config_d['model']['n_filters']
    config.n_mels = config_d['tfrecords']['n_mels']
    config.n_output_neurons = len(tags)
    config.path = config_path
    config.plateau_min_d = config_d['config']['reduce_lr_plateau_min_delta']
    config.plateau_patience = config_d['config']['reduce_lr_plateau_patience']
    config.shuffle = config_d['config']['shuffle']
    config.shuffle_buffer = config_d['config']['shuffle_buffer_size']
    config.split = config_d['config']['split']
    config.sr = config_d['tfrecords']['sample_rate']
    config.tags = lastfm.vec_tag_to_tag_num(list(tags))
    config.tags_to_merge = lastfm.tag_to_tag_num(config_d['tags']['merge']) if config_d['tags']['merge'] is not None else None
    config.tot_tags = config_d['tfrecords']['n_tags']
    config.window_len = config_d['config']['window_length']
    config.window_random = config_d['config']['window_extract_randomly']
    config.log_dir = config_d['config']['log_dir']
    config.checkpoint_dir = config_d['config']['checkpoint_dir']

    # create config namespace for the optimizer (will be used by get_optimizer() in order to allow max flexibility)
    config_optim = argparse.Namespace()
    config_optim.class_name = config_d['optimizer'].pop('name')
    config_optim.config = config_d['optimizer']
    
    return config, config_optim
            
def train(train_dataset, valid_dataset, frontend, strategy, config, config_optim, epochs, resume_time=None, update_freq=1, analyse_trace=True):

    log_dir = os.path.join(os.path.expanduser(config.log_dir), datetime.datetime.now().strftime("%y%m%d-%H%M")) # to save training metrics (to access using tensorboard)
    checkpoint_dir = os.path.join(os.path.join(os.path.expanduser(config.checkpoint_dir), frontend + '_' + datetime.datetime.now().strftime("%y%m%d-%H%M"))) # to save model checkpoints
    
    with strategy.scope():
        # build model
        model = projectname.build_model(frontend, num_output_neurons=config.n_output_neurons, num_units=config.n_dense_units, num_filts=config.n_filters, y_input=config.n_mels)
        
        # initialise loss, optimizer and metrics
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
            checkpoint_dir = os.path.join(os.path.expanduser(config.checkpoint_dir), resume_time) # keep saving checkpoints in the 'old' folder
            
            # try to load checkpoint
            chkp = tf.train.latest_checkpoint(checkpoint_dir)
            if chkp:
                tf.print("Checkpoint file {} found. Restoring...".format(chkp))
                checkpoint.restore(chkp)
                tf.print("Checkpoint restored.")
                prev_epoch = int(chkp.split('-')[-1])-1
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
        def distributed_train_body(entry, epoch):
            num_batches = 0 
            for entry in train_dataset:
                strategy.experimental_run_v2(train_step, args=(entry, ))
                num_batches += 1
                # print metrics after each iteration
                if tf.equal(num_batches % update_freq, 0):
                    tf.print('{}/Unknown - loss: {:8.6f} - AUC-ROC {:6.5f} - AUC-PR {:6.5f}'.format(num_batches, train_mean_loss.result(), train_metrics_1.result(), train_metrics_2.result()))
                    with train_summary_writer.as_default():
                        tf.summary.scalar('batch_AUC-ROC', train_metrics_1.result(), step=optimizer.iterations)
                        tf.summary.scalar('batch_AUC-PR', train_metrics_2.result(), step=optimizer.iterations)
                        tf.summary.scalar('batch_loss', train_mean_loss.result(), step=optimizer.iterations)
                        train_summary_writer.flush()

        @tf.function
        def distributed_val_body(entry):
            for entry in valid_dataset:
                strategy.experimental_run_v2(valid_step, args=(entry, ))
        
        max_metric = -200 # for early stopping

        # loop
        for epoch in tf.range(prev_epoch+1, epochs, dtype=tf.int64):
            start_time = time.time()
            tf.print()
            tf.print()
            tf.print('Epoch {}/{}'.format(epoch, epochs))

            tf.summary.trace_on(graph=False, profiler=True)
            
            distributed_train_body(train_dataset, epoch)
            
            # write metrics on tensorboard after each epoch
            with train_summary_writer.as_default():
                tf.summary.scalar('epoch_AUC-ROC', train_metrics_1.result(), step=epoch)
                tf.summary.scalar('epoch_AUC-PR', train_metrics_2.result(), step=epoch)
                tf.summary.scalar('epoch_loss', train_mean_loss.result(), step=epoch)
                train_summary_writer.flush()
                
            # print progress
            tf.print('Epoch {}: loss {:8.6f} - AUC-ROC {:6.5f} - AUC-PR {:6.5f}'.format(epoch, train_mean_loss.result(), train_metrics_1.result(), train_metrics_2.result()), end=' ')
            
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
                with val_summary_writer.as_default():
                    tf.summary.scalar('epoch_AUC-ROC', val_metrics_1.result(), step=epoch)
                    tf.summary.scalar('epoch_AUC-PR', val_metrics_2.result(), step=epoch)
                    tf.summary.scalar('epoch_loss', val_loss.result(), step=epoch)
                    val_summary_writer.flush()

                tf.print('- val_AUC-ROC {:6.5f} - val_AUC_PR {:6.5f}'.format(val_metrics_1.result(), val_metrics_2.result()))
                
                # reset validation metrics after each epoch
                val_metrics_1.reset_states()
                val_metrics_2.reset_states()
                val_loss.reset_states()
                
                # early stopping
                if (config.early_stop_min_d) or (config.early_stop_patience):
                    
                    if not config.early_stop_min_d:
                        config.early_stop_min_d = 0
                   
                    if not config.early_stop_patience:
                        config.early_stop_patience = 1
                    
                    if os.path.isfile(os.path.join(checkpoint_dir, 'early_stopping.npy')):
                        cumerror = int(np.load(os.path.join(checkpoint_dir, 'early_stopping.npy')))
                    
                    if val_metrics_2 > (max_metric + config.early_stop_min_d):
                        max_metric = val_metrics_2
                        cumerror = 0
                        np.save(os.path.join(checkpoint_dir, 'early_stopping.npy'), cumerror)
                    else:
                        cumerror += 1
                        tf.print('Epoch {}: no significant improvements ({}/{})'.format(epoch, epochs, cumerror, config.early_stop_patience))
                        np.save(os.path.join(checkpoint_dir, 'early_stopping.npy'), cumerror)
                        if cumerror == config.early_stop_patience:
                            tf.print('Epoch {}: stopping')
                            break
                    
            elif (config.early_stop_min_d) or (config.early_stop_patience):
                raise RuntimeError('EarlyStopping requires a validation dataset')

            checkpoint_path = os.path.join(checkpoint_dir, 'epoch-'+epoch, 'mymodel.h5')
            checkpoint.save(checkpoint_path)
            tf.print('Epoch {}: saving model to {}'.format(checkpoint_path))

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
    parser.add_argument("-v", "--verbose", choices=['0', '1', '2', '3'], help="verbose mode", default='2')

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
                                                                                shuffle=config.shuffle, buffer_size=config.shuffle_buffer, 
                                                                                num_tags=config.tot_tags, window_size=config.window_len, random=config.window_random, 
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

# def main(tfrecords_dir, frontend, config_dir, resume_time=None,
#          split=(70, 10, 20), epochs=5, sample_rate=16000, batch_size=32,
#          cycle_length=2, validation=True, shuffle=True, buffer_size=10000,
#          window_size=15, random=False, with_tags=None, merge_tags=None,
#          log_dir = '/srv/data/urop/log_aden/', checkpoint_dir='/srv/data/urop/model_aden',
#          with_tids=None, analyse_trace=False, config.early_stop_min_d=None,
#          config.early_stop_patience=None):
   
#     '''Combines data input pipeline, networks, train and validation loops to 
#         perform model training.

#     Parameters
#     ----------
#     tfrecords_dir: str
#         The directory of where the tfrecord files are stored.
        
#     frontend: str
#         'waveform' or 'log-mel-spectrogram', indicating the format of the
#         audio inputs contained in the tfrecord files.
        
#     config_dir: str
#         The directory (config.json) or path of where the json file (contains 
#         training and dataset configuration info) created in projectname.py 
#         is stored.
        
#     checkpoint: str
#         The time denoted in the latest checkpoint file of format 'YYMMDD-hhmmss'.
#         You may find out the time by viewing the folder name stored under the 
#         checkpoint_dir, e.g. log-mel-spectrogram_20190823-000120, then checkpoint
#         should be equal to '20190823-000120'.

#     split: tuple (a tuple of three integers)
#         Specifies the train/validation/test percentage to use when selecting 
#         the .tfrecord files.
        
#     epochs: int
#         Number of epochs.
        
#     sample_rate: int
#         The sampling rate of the audio data, this should be consistent with
#         the rate used to generate the tfrecord files.

#     batch_size: int
#         Specifies the dataset batch_size.
        
#     cycle_length: int
#         Controls the number of input elements that are processed concurrently.

#     validation: bool
#         If True, validation is performed within each epoch.
        
#     shuffle: bool
#         If True, shuffles the dataset with buffer size = buffer_size.

#     buffer_size: int
#         If shuffle is True, sets the shuffle buffer size.

#     window_size: int
#         Specifies the desired window length (in seconds) for the audio data
#         in the datasets.

#     random: bool
#         Specifies how the window is to be extracted. If True, slices 
#         the window randomly (default is pick from the middle).

#     with_tags: list
#         If not None, contains the tags to use.

#     merge_tags: list
#         If not None, contains the lists of tags to be merged together 
#         (only applies if with_tags is specified).
        
#     log_dir: str
#         The directory where the tensorboard data (profiling, AUC_PR, AUC_ROC, 
#         loss logging) are stored.
        
#     checkpoint_dir: str
#         The directory where the Checkpoints files from each epoch will be 
#         stored. Note that the actual files will be stored under a subfolder
#         based on the frontend.
        
#     with_tids: str
#         If not None, contains the tids to be trained on.
        
#     analyse_trace: bool
#         If True, the trace information (profiling in tensorboard) is stored
#         for each epoch.
        
#     config.early_stop_min_d: float
#         The validation PR-AUC in an epoch is greater than the sum of max 
#         validation PR-AUC and config.early_stop_min_d (when 
#         config.early_stop_patience=0) if and only if the validation is counted 
#         as an improvement. If this is not None, early stopping will be 
#         automatically enabled wih default config.early_stop_patience=1
    
#     config.early_stop_patience: int
#         The number of consecutive 'no improvement' epochs to trigger early
#         stopping to stop the training. If this is not None, early stopping
#         will be automatically enabled with default config.early_stop_min_d=0.
    
#     '''
    
#     #initialise configuration
#     if not os.path.isfile(config_dir):
#         config_dir = os.path.join(os.path.normpath(config_dir), 'config.json')
        
#     with open(config_dir) as f:
#         file = json.load(f)
        
#     num_tags = file['dataset_specs']['n_tags']
#     y_input = file['dataset_specs']['n_mels']
#     lr = file['training_options']['lr']
#     num_units = file['training_options']['n_dense_units']
#     num_filt = file['training_options']['n_filters']
#     num_output_neurons = file['training_options']['n_output_neurons']

#     strategy = tf.distribute.MirroredStrategy()
    
#     train_dataset, val_dataset = \
#     projectname_input.generate_datasets_from_dir(tfrecords_dir=tfrecords_dir,
#                                                  audio_format=frontend, 
#                                                  split=split, 
#                                                  sample_rate=sample_rate,
#                                                  batch_size=batch_size,
#                                                  cycle_length=cycle_length,
#                                                  shuffle=shuffle,
#                                                  buffer_size=buffer_size, 
#                                                  window_size=window_size, 
#                                                  random=random,
#                                                  with_tags=with_tags, 
#                                                  merge_tags=merge_tags,
#                                                  with_tids=with_tids, 
#                                                  num_tags=num_tags,
#                                                  epochs=1,
#                                                  as_tuple=False)[:2]

#     train_dataset = strategy.experimental_distribute_dataset(train_dataset)
#     valid_dataset = strategy.experimental_distribute_dataset(val_dataset)
    
#     # should we use this? Should this be hard coded into config.json?
#     if with_tags:
#         num_output_neurons = len(with_tags)
#         if merge_tags:
#             num_output_neurons = num_output_neurons - len(merge_tags)
    
#     train(frontend=frontend, 
#           train_dataset=train_dataset, 
#           strategy=strategy, 
#           resume_time=resume_time,
#           valid_dataset=valid_dataset, 
#           validation=validation,  
#           epochs=epochs, 
#           num_output_neurons=num_output_neurons, 
#           y_input=y_input, 
#           num_units=num_units, 
#           num_filt=num_filt, 
#           global_batch_size=batch_size,
#           lr=lr, 
#           log_dir=log_dir, 
#           checkpoint_dir=checkpoint_dir,
#           analyse_trace=analyse_trace,
#           config.early_stop_min_d=config.early_stop_min_d,
#           config.early_stop_patience=config.early_stop_patience)
