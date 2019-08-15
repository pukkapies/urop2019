'''
TODO LIST:
- add printing
- fix tensorboard and make it run on Boden (create graph, profile)
- evaluation method (include test phase and convert tag_num to tag)
'''

import sys
sys.path.insert(0, 'C://Users/hcw10/UROP2019')

import numpy as np
import tensorflow as tf
import model as Model
from datetime import datetime
import os
import time
import projectName_input
import json


def train(frontend_mode, train_dist_dataset, strategy, val_dist_dataset=None, validation=True, 
          num_epochs=10, numOutputNeurons=155, y_input=96, num_units=1024, 
          num_filt=32, lr=0.001, log_dir = 'logs/trial1/'):
    
    with strategy.scope():
        #import model
        model = Model.build_model(frontend_mode=frontend_mode,
                                  numOutputNeurons=numOutputNeurons,
                                  y_input=y_input, num_units=num_units, 
                                  num_filt=num_filt)
        
        #initialise loss, optimizer, metric
        optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)

        train_AUC = tf.keras.metrics.AUC(name='train_AUC', dtype=tf.float32)
        train_loss = tf.keras.metrics.Mean(name='training_loss', dtype=tf.float32)

        loss = tf.keras.losses.MeanSquaredError()

        if validation:
            val_AUC = tf.keras.metrics.AUC(name='val_AUC', dtype=tf.float32)
            # TODO: val loss?
            # val_loss = tf.keras.metrics.Mean(name='val_loss', dtype=tf.float32)

        #in case of keyboard interrupt during previous training
        tf.summary.trace_off()
        
        # setting up summary writers
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')

        train_log_dir = log_dir + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        
        prof_log_dir = log_dir + current_time + '/prof'
        prof_summary_writer = tf.summary.create_file_writer(prof_log_dir)

        if validation:
            val_log_dir = log_dir + current_time + '/val'
            val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        # fucntions needs to be defined within the strategy scope
        def train_step(entry):
            audio_batch, label_batch = entry['audio'], entry['tags']

            with tf.GradientTape() as tape:
                logits = model(audio_batch) # TODO: training=True????

                loss_value = loss(label_batch, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            # UPDATE LOSS METRIC
            train_AUC.update_state(label_batch, logits)
            train_loss.update_state(loss_value)

        def val_step(entry):
            audio_batch, label_batch = entry['audio'], entry['tags']

            logits = model(audio_batch)
            # TODO: record loss for val dataset?
            # loss_value = loss(label_batch, logits)
            # val_loss.update_state(loss_value)
            val_AUC.update_state(label_batch, logits)

        @tf.function 
        def distributed_train_body(dist_dataset):

            for step, entry in dist_dataset.enumerate()
                strategy.experimental_run_v2(train_step, args=(entry,))

        @tf.function
        def distributed_val_body(dist_dataset):
            tf.keras.backend.set_learning_phase(0)

            for entry in dist_dataset:
                strategy.experimental_run_v2(val_step, args=(entry,))

            tf.keras.backend.set_learning_phase(1)
    
        #epoch loop
        for epoch in range(num_epochs):
            start_time = time.time()
            tf.print('Epoch {}'.format(epoch))

            tf.summary.trace_on(graph=False, profiler=True)

            distributed_train_body(train_dist_dataset)            

            # print progress
            tf.print('Epoch', epoch,  ': loss', train_loss.result(), '; AUC', train_AUC.result())
            # log to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('AUC', train_AUC.result(), step=epoch)
                tf.summary.scalar('Loss', train_loss.result(), step=epoch)

            #print progress
            tf.print('Epoch {} --training done'.format(epoch))

            # tensorboard export profiling and record train AUC and loss
            with prof_summary_writer.as_default():   
                tf.summary.trace_export(
                        name="trace", 
                        step=epoch, profiler_outdir=os.path.normpath(prof_log_dir)) 

            train_AUC.reset_states()
            train_loss().reset_states()

            if validation:
               distributed_val_body(val_dist_dataset) 

                with val_summary_writer.as_default():
                    tf.summary.scalar('AUC', val_AUC.result(), step=epoch)
                tf.print('Val- Epoch', epoch, ': AUC', val_AUC.result())
                
                # reset val metric per epoch
                val_AUC.reset_states()

            #report time
            time_taken = time.time()-start_time
            tf.print('Time taken for epoch {}: {}s'.format(epoch, time_taken))

def main(tfrecord_dir, frontend_mode, config_dir, train_val_test_split=(70, 10, 20),
         batch_size=32, validation=True, shuffle=True, buffer_size=10000, 
         window_length=15, random=False, with_tags=None,
         log_dir = 'logs/trial1/', with_tids=None, num_epochs=5):
    
    #initialise configuration
    if not os.path.isfile(config_dir):
        config_dir = os.path.join(os.path.normpath(config_dir), 'config.txt')
        
    with open(config_dir) as f:
        file = json.load(f)
        
    numOutputNeurons = file['data_params']['n_tags']
    y_input = file['data_params']['n_mels']
    lr = file['train_params']['lr']
    num_units = file['train_params']['n_dense_units']
    num_filt = file['train_params']['n_filters']

    strategy = tf.distribute.MirroredStrategy()

    train_dataset, val_dataset = \
    generate_datasets(tfrecord_dir=tfrecord_dir, audio_format=frontend_mode, 
                      train_val_test_split=train_val_test_split, 
                      which = [True, True, False],
                      batch_size=batch_size, shuffle=shuffle, 
                      buffer_size=buffer_size, window_length=window_length, 
                      random=random, with_tags=with_tags, with_tids=with_tids, 
                      num_epochs=num_epochs)

    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)
    
    train(frontend_mode=frontend_mode, train_dist_dataset=train_dist_dataset, 
          strategy=strategy, val_dist_dataset=val_dist_dataset, validation=validation,  
          num_epochs=num_epochs, numOutputNeurons=numOutputNeurons, 
          y_input=y_input, num_units=num_units, num_filt=num_filt, 
          lr=lr, log_dir=log_dir)
