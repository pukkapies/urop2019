'''
TODO LIST:
- add printing
- fix tensorboard and make it run on Boden (create graph, profile)
- evaluation method (include test phase and convert tag_num to tag)
'''

import sys
import os
import time
import json
from datetime import datetime
sys.path.insert(0, 'C://Users/hcw10/UROP2019')

import numpy as np
import tensorflow as tf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')))
import projectname as Model
import train_cpu


def train(frontend_mode, train_dist_dataset, strategy, val_dist_dataset=None, validation=True, 
          num_epochs=10, numOutputNeurons=155, y_input=96, num_units=1024, global_batch_size=32,
          num_filt=32, lr=0.001, log_dir = 'logs/trial1/', model_dir='/srv/data/urop/model'):
    with strategy.scope():
        #import model
        model = Model.build_model(frontend_mode=frontend_mode,
                                  num_output_neurons=numOutputNeurons,
                                  y_input=y_input, num_units=num_units, 
                                  num_filt=num_filt)
        
        #initialise loss, optimizer, metric
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        train_AUC = tf.keras.metrics.AUC(name='train_AUC', dtype=tf.float32)

        loss_obj = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

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

                loss = tf.nn.compute_average_loss(loss_obj(label_batch, logits), global_batch_size=global_batch_size)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            train_AUC.update_state(label_batch, logits)
            return loss


        def val_step(entry):
            audio_batch, label_batch = entry['audio'], entry['tags']

            logits = model(audio_batch)
            # TODO: record loss for val dataset?
            # loss_value = loss(label_batch, logits)
            # val_loss.update_state(loss_value)
            val_AUC.update_state(label_batch, logits)

        @tf.function 
        def distributed_train_body(dist_dataset):
            total_loss = 0.0
            num_batches = 0

            for entry in dist_dataset:
                per_replica_losses = strategy.experimental_run_v2(train_step, args=(entry,))
                total_loss += strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                num_batches += 1

            return total_loss / tf.cast(num_batches, tf.float32)

        @tf.function
        def distributed_val_body(dist_dataset):
            tf.keras.backend.set_learning_phase(0)

            for entry in dist_dataset:
                strategy.experimental_run_v2(val_step, args=(entry,))

            tf.keras.backend.set_learning_phase(1)


        # setting up checkpoints
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        latest_checkpoint_file = tf.train.latest_checkpoint(model_dir)
        if latest_checkpoint_file:
            tf.print('Checkpoint file {} found, restoring'.format(latest_checkpoint_file))
            checkpoint.restore(latest_checkpoint_file)
            tf.print('Loading from checkpoint file completed')

        #epoch loop
        for epoch in range(num_epochs):
            start_time = time.time()
            tf.print('Epoch {}'.format(epoch))

            tf.summary.trace_on(graph=False, profiler=True)

            loss = distributed_train_body(train_dist_dataset)            

            # print progress
            tf.print('Epoch', epoch,  ': loss', loss, '; AUC', train_AUC.result())
            # log to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('AUC', train_AUC.result(), step=epoch)
                tf.summary.scalar('Loss', loss, step=epoch)

            #print progress
            tf.print('Epoch {} --training done'.format(epoch))

            # tensorboard export profiling and record train AUC and loss
            with prof_summary_writer.as_default():   
                tf.summary.trace_export(
                        name="trace", 
                        step=epoch, profiler_outdir=os.path.normpath(prof_log_dir)) 

            train_AUC.reset_states()

            if validation:
                distributed_val_body(val_dist_dataset) 

                with val_summary_writer.as_default():
                    tf.summary.scalar('AUC', val_AUC.result(), step=epoch)
                tf.print('Val- Epoch', epoch, ': AUC', val_AUC.result())
                
                # reset val metric per epoch
                val_AUC.reset_states()

            checkpoint.save(checkpoint_prefix)
            checkpoint_path = os.path.join(model_dir, 'epoch_{}.ckpt'.format(epoch))
            saved_path = checkpoint.save(checkpoint_path)
            tf.print('Saving model as TF checkpoint: {}'.format(saved_path))

            #report time
            time_taken = time.time()-start_time
            tf.print('Time taken for epoch {}: {}s'.format(epoch, time_taken))

def main(tfrecord_dir, frontend_mode, config_dir, train_val_test_split=(70, 10, 20),
         batch_size=32, validation=True, shuffle=True, buffer_size=10000, 
         window_length=15, random=False, with_tags=None,
         log_dir = 'logs/trial1/', with_tids=None, num_epochs=5):
    
    #initialise configuration
    if not os.path.isfile(config_dir):
        config_dir = os.path.join(os.path.normpath(config_dir), 'config.json')
        
    with open(config_dir) as f:
        file = json.load(f)
        
    numOutputNeurons = file['dataset_specs']['n_tags']
    y_input = file['dataset_specs']['n_mels']
    lr = file['training_options']['lr']
    num_units = file['training_options']['n_dense_units']
    num_filt = file['training_options']['n_filters']

    strategy = tf.distribute.MirroredStrategy()

    train_dataset, val_dataset = \
    train_cpu.generate_datasets(tfrecord_dir=tfrecord_dir, audio_format=frontend_mode, 
                      train_val_test_split=train_val_test_split, 
                      which = [True, True, False],
                      batch_size=batch_size, shuffle=shuffle, 
                      buffer_size=buffer_size, window_length=window_length, 
                      random=random, with_tags=with_tags, with_tids=with_tids, 
                      num_epochs=1)

    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)
    
    train(frontend_mode=frontend_mode, train_dist_dataset=train_dist_dataset, 
          strategy=strategy, val_dist_dataset=val_dist_dataset, validation=validation,  
          num_epochs=num_epochs, numOutputNeurons=numOutputNeurons, 
          y_input=y_input, num_units=num_units, num_filt=num_filt, 
          lr=lr, log_dir=log_dir)

if __name__ == '__main__':
    main('/srv/data/urop/tfrecords-waveform', 'waveform', '/home/calle/config.json')
