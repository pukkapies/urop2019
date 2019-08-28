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

Functions
---------
- train
    Compile model with optimisers, loss, and train with customed training loops 
    and validation loops.
    
- main
    Combine data input pipeline, networks, train and validation loops to 
    perform model training.
     (on Boden: export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64")
'''

import argparse
import datetime
import gc
import json
import os
import time
import sys

import numpy as np
import tensorflow as tf

import projectname
import projectname_input
from modules.query_lastfm import LastFm
            
def train(frontend, train_dist_dataset, strategy, checkpoint_path=None, val_dist_dataset=None, validation=True, 
          num_epochs=10, num_output_neurons=155, y_input=96, num_units=1024, global_batch_size=32,
          num_filt=32, lr=0.001, log_dir = 'logs/trial1/', model_dir='/srv/data/urop/model',
          analyse_trace=False, early_stopping_min_delta=None, early_stopping_patience=None):

    log_dir = os.path.join(os.path.expanduser(log_dir), frontend, datetime.datetime.now().strftime("%y%m%d-%H%M")) # to access training scalars using tensorboard
    
    with strategy.scope():

        model = projectname.build_model(frontend, num_output_neurons=num_output_neurons, y_input=y_input, num_units=num_units, num_filt=num_filt)
        
        # initialise loss, optimizer, metrics
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss_obj = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        
        train_AUC_ROC = tf.keras.metrics.AUC(curve='ROC', name='train_AUC-ROC', dtype=tf.float32)
        train_AUC_PR = tf.keras.metrics.AUC(curve='PR', name='train_AUC-PR', dtype=tf.float32)
        train_mean_loss = tf.keras.metrics.Mean(name='train_mean_loss', dtype=tf.float32)

        # setting up checkpoints
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        prev_epoch = -1
        
        # resume
        if checkpoint_path is None:
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
        else:
            latest_checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
            if latest_checkpoint_file:
                tf.print("Checkpoint file {} found. Restoring...".format(latest_checkpoint_file))
                checkpoint.restore(latest_checkpoint_file)
                tf.print("Checkpoint restored.")
                prev_epoch = int(latest_checkpoint_file.split('-')[-1])-1
                log_dir = checkpoint_path # use checkpoint log_dir
            else:
                tf.print("Checkpoint file not found!")
                return
        
        tf.summary.trace_off() # in case of previous keyboard interrupt
        
        # setting up summary writers
        train_log_dir = os.path.join(log_dir, 'train')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        
        if validation:
            val_log_dir = os.path.join(log_dir, 'validation')
            val_summary_writer = tf.summary.create_file_writer(val_log_dir)
            val_metrics_1 = tf.keras.metrics.AUC(curve = 'ROC', name='val_AUC-ROC', dtype=tf.float32)
            val_metrics_2 = tf.keras.metrics.AUC(curve = 'PR', name='val_AUC-PR', dtype=tf.float32)
            val_loss = tf.keras.metrics.Mean(name='val_loss', dtype=tf.float32)
        
        if analyse_trace: # make sure the environment variable LD_LIBRARY_PATH is set up properly!!
            prof_log_dir = log_dir + log_time + '/prof'
            prof_summary_writer = tf.summary.create_file_writer(prof_log_dir)
        
        # rescale loss
        def compute_loss(labels, predictions):
            per_example_loss = loss_obj(labels, predictions)
            return per_example_loss/global_batch_size
        
        def train_step(entry):
            audio_batch, label_batch = entry['audio'], entry['tags']
            with tf.GradientTape() as tape:
                logits = model(audio_batch)
                loss = compute_loss(label_batch, logits)
            variables = model.trainable_variables
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            train_AUC_ROC.update_state(label_batch, logits)
            train_AUC_PR.update_state(label_batch, logits)
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
            
            for entry in train_dist_dataset:
                strategy.experimental_run_v2(train_step, args=(entry,))
                
                num_batches += 1
                
                if tf.equal(num_batches % 10, 0):
                    tf.print('Epoch', epoch,'; Step', num_batches, '; loss', train_mean_loss.result(), '; AUC-ROC', train_AUC_ROC.result(), ';AUC-PR', train_AUC_PR.result())
                    
                    # write metrics on tensorboard after each iteration
                    with train_summary_writer.as_default():
                        tf.summary.scalar('iter_AUC-ROC', train_AUC_ROC.result(), step=optimizer.iterations)
                        tf.summary.scalar('iter_AUC-PR', train_AUC_PR.result(), step=optimizer.iterations)
                        tf.summary.scalar('iter_loss', train_mean_loss.result(), step=optimizer.iterations)
                        train_summary_writer.flush()

        @tf.function
        def distributed_val_body(entry):
            for entry in val_dist_dataset:
                strategy.experimental_run_v2(valid_step, args=(entry, ))
        
        max_metric = -200 # for early stopping

        # loop
        for epoch in tf.range(prev_epoch+1, num_epochs, dtype=tf.int64):
            start_time = time.time()
            tf.print('Epoch {}'.format(epoch))

            tf.summary.trace_on(graph=False, profiler=True)
            
            distributed_train_body(train_dist_dataset, epoch)
            
            # write metrics on tensorboard after each epoch
            with train_summary_writer.as_default():
                tf.summary.scalar('epoch_AUC-ROC', train_AUC_ROC.result(), step=epoch)
                tf.summary.scalar('epoch_AUC-PR', train_AUC_PR.result(), step=epoch)
                tf.summary.scalar('epoch_loss', train_mean_loss.result(), step=epoch)
                train_summary_writer.flush()
                
            # print progress
            tf.print('Epoch', epoch, ': loss', train_mean_loss.result(), '; AUC-ROC', train_AUC_ROC.result(), '; AUC-PR', train_AUC_PR.result())

            # print progress
            tf.print('Epoch {} --training done\n'.format(epoch))
            
            train_AUC_ROC.reset_states()
            train_AUC_PR.reset_states()
            train_mean_loss.reset_states()

            # export profiling and write validation metrics on tensorboard
            if analyse_trace:
                with prof_summary_writer.as_default():   
                    tf.summary.trace_export(name="trace", 
                                            step=epoch, 
                                            profiler_outdir=os.path.normpath(prof_log_dir)) 

            if validation:
                
                distributed_val_body(val_dist_dataset)
                
                with val_summary_writer.as_default():
                    tf.summary.scalar('epoch_AUC-ROC', val_metrics_1.result(), step=epoch)
                    tf.summary.scalar('epoch_AUC-PR', val_metrics_2.result(), step=epoch)
                    tf.summary.scalar('epoch_loss', val_loss.result(), step=epoch)
                    val_summary_writer.flush()

                tf.print('Val- Epoch', epoch, ': AUC-ROC', val_metrics_1.result(), '; AUC-PR', val_metrics_2.result())
                
                # reset validation metrics after each epoch
                val_metrics_1.reset_states()
                val_metrics_2.reset_states()
                val_loss.reset_states()
                
                # early stopping
                if (early_stopping_min_delta) or (early_stopping_patience):
                    if not early_stopping_min_delta:
                        early_stopping_min_delta = 0.
                    if not early_stopping_patience:
                        early_stopping_patience = 1
                    
                    if os.path.isfile(os.path.join(log_dir, 'early_stopping.npy')):
                        cumerror = int(np.load(os.path.join(log_dir, 'early_stopping.npy')))
                    
                    if val_metrics_2 > (max_metric + early_stopping_min_delta):
                        max_metric = val_metrics_2
                        cumerror = 0
                        np.save(os.path.join(log_dir, 'early_stopping.npy'), cumerror)
                    else:
                        cumerror += 1
                        tf.print('No improvements - {}/{}'.format(cumerror, early_stopping_patience))
                        np.save(os.path.join(log_dir, 'early_stopping.npy'), cumerror)
                        if cumerror == early_stopping_patience:
                            tf.print('Early stopping criteria met. Stopping...')
                            break
                    #TODO: record early stopping progress in case crushes
                    
            elif (early_stopping_min_delta) or (early_stopping_patience):
                tf.print('Early stopping requires a validation dataset...')

            checkpoint_path = os.path.join(log_dir, epoch)
            saved_path = checkpoint.save(checkpoint_path)
            tf.print('Saving checkpoint: {}'.format(saved_path))

            #report time
            time_taken = time.time()-start_time
            tf.print('Time taken for epoch {}: {} s'.format(epoch, time_taken))
            
            tf.keras.backend.clear_session()
            gc.collect()

def main(tfrecords_dir, frontend, config_dir, checkpoint_path=None,
         split=(70, 10, 20), num_epochs=5, sample_rate=16000, batch_size=32,
         cycle_length=2, validation=True, shuffle=True, buffer_size=10000,
         window_size=15, random=False, with_tags=None, merge_tags=None,
         log_dir = '/srv/data/urop/log_aden/', model_dir='/srv/data/urop/model_aden',
         with_tids=None, analyse_trace=False, early_stopping_min_delta=None,
         early_stopping_patience=None):
   
    '''Combines data input pipeline, networks, train and validation loops to 
        perform model training.

    Parameters
    ----------
    tfrecords_dir: str
        The directory of where the tfrecord files are stored.
        
    frontend: str
        'waveform' or 'log-mel-spectrogram', indicating the format of the
        audio inputs contained in the tfrecord files.
        
    config_dir: str
        The directory (config.json) or path of where the json file (contains 
        training and dataset configuration info) created in projectname.py 
        is stored.
        
    checkpoint: str
        The time denoted in the latest checkpoint file of format 'YYMMDD-hhmmss'.
        You may find out the time by viewing the folder name stored under the 
        model_dir, e.g. log-mel-spectrogram_20190823-000120, then checkpoint
        should be equal to '20190823-000120'.

    split: tuple (a tuple of three integers)
        Specifies the train/validation/test percentage to use when selecting 
        the .tfrecord files.
        
    num_epochs: int
        Number of epochs.
        
    sample_rate: int
        The sampling rate of the audio data, this should be consistent with
        the rate used to generate the tfrecord files.

    batch_size: int
        Specifies the dataset batch_size.
        
    cycle_length: int
        Controls the number of input elements that are processed concurrently.

    validation: bool
        If True, validation is performed within each epoch.
        
    shuffle: bool
        If True, shuffles the dataset with buffer size = buffer_size.

    buffer_size: int
        If shuffle is True, sets the shuffle buffer size.

    window_size: int
        Specifies the desired window length (in seconds) for the audio data
        in the datasets.

    random: bool
        Specifies how the window is to be extracted. If True, slices 
        the window randomly (default is pick from the middle).

    with_tags: list
        If not None, contains the tags to use.

    merge_tags: list
        If not None, contains the lists of tags to be merged together 
        (only applies if with_tags is specified).
        
    log_dir: str
        The directory where the tensorboard data (profiling, AUC_PR, AUC_ROC, 
        loss logging) are stored.
        
    model_dir: str
        The directory where the Checkpoints files from each epoch will be 
        stored. Note that the actual files will be stored under a subfolder
        based on the frontend.
        
    with_tids: str
        If not None, contains the tids to be trained on.
        
    analyse_trace: bool
        If True, the trace information (profiling in tensorboard) is stored
        for each epoch.
        
    early_stopping_min_delta: float
        The validation PR-AUC in an epoch is greater than the sum of max 
        validation PR-AUC and early_stopping_min_delta (when 
        early_stopping_patience=0) if and only if the validation is counted 
        as an improvement. If this is not None, early stopping will be 
        automatically enabled wih default early_stopping_patience=1
    
    early_stopping_patience: int
        The number of consecutive 'no improvement' epochs to trigger early
        stopping to stop the training. If this is not None, early stopping
        will be automatically enabled with default early_stopping_min_delta=0.
    
    '''
    
    #initialise configuration
    if not os.path.isfile(config_dir):
        config_dir = os.path.join(os.path.normpath(config_dir), 'config.json')
        
    with open(config_dir) as f:
        file = json.load(f)
        
    num_tags = file['dataset_specs']['n_tags']
    y_input = file['dataset_specs']['n_mels']
    lr = file['training_options']['lr']
    num_units = file['training_options']['n_dense_units']
    num_filt = file['training_options']['n_filters']
    num_output_neurons = file['training_options']['n_output_neurons']

    strategy = tf.distribute.MirroredStrategy()
    
    train_dataset, val_dataset = \
    projectname_input.generate_datasets_from_dir(tfrecords_dir=tfrecords_dir,
                                                 audio_format=frontend, 
                                                 split=split, 
                                                 sample_rate=sample_rate,
                                                 batch_size=batch_size,
                                                 cycle_length=cycle_length,
                                                 shuffle=shuffle,
                                                 buffer_size=buffer_size, 
                                                 window_size=window_size, 
                                                 random=random,
                                                 with_tags=with_tags, 
                                                 merge_tags=merge_tags,
                                                 with_tids=with_tids, 
                                                 num_tags=num_tags,
                                                 num_epochs=1,
                                                 as_tuple=False)[:2]

    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)
    
    # should we use this? Should this be hard coded into config.json?
    if with_tags:
        num_output_neurons = len(with_tags)
        if merge_tags:
            num_output_neurons = num_output_neurons - len(merge_tags)
    
    train(frontend=frontend, 
          train_dist_dataset=train_dist_dataset, 
          strategy=strategy, 
          checkpoint_path=checkpoint_path,
          val_dist_dataset=val_dist_dataset, 
          validation=validation,  
          num_epochs=num_epochs, 
          num_output_neurons=num_output_neurons, 
          y_input=y_input, 
          num_units=num_units, 
          num_filt=num_filt, 
          global_batch_size=batch_size,
          lr=lr, 
          log_dir=log_dir, 
          model_dir=model_dir,
          analyse_trace=analyse_trace,
          early_stopping_min_delta=early_stopping_min_delta,
          early_stopping_patience=early_stopping_patience)
    
    
    
if __name__ == '__main__':

    fm = LastFm('/srv/data/urop/clean_lastfm.db') 
    tags = fm.popularity().tag.to_list()[:50]
    with_tags = [fm.tag_to_tag_num(tag) for tag in tags]

    CONFIG_FOLDER = '/home/davide'

    main('/srv/data/urop/tfrecords-waveform', 'waveform', 
                CONFIG_FOLDER, split=(58, 1, 1),  shuffle=True, batch_size=64,
                buffer_size=1000, random=True, log_dir='/srv/data/urop/model/logs/',
                with_tags=with_tags, num_epochs=10, model_dir='/srv/data/urop/model/')
