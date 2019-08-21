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

import tensorflow as tf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')))
import projectname as Model
import train_cpu



def train(frontend_mode, train_dist_dataset, strategy, val_dist_dataset=None, validation=True, 
          num_epochs=10, num_output_neurons=155, y_input=96, num_units=1024, global_batch_size=32,
          num_filt=32, lr=0.001, log_dir = 'logs/trial1/', model_dir='/srv/data/urop/model',
          analyse_trace=False):
    '''Trains model, see doc on main() for more details.'''
    
    
    with strategy.scope():
        #import model
        print('Building Model')
        model = Model.build_model(frontend_mode=frontend_mode,
                                  num_output_neurons=num_output_neurons,
                                  y_input=y_input, num_units=num_units, 
                                  num_filt=num_filt)
        
        #s = 20 * 3000 // global_batch_size
        
        #initialise loss, optimizer, metric
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        #learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
        #optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        loss_obj = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        train_ROC_AUC = tf.keras.metrics.AUC(curve='ROC', name='train_ROC_AUC', dtype=tf.float32)
        train_PR_AUC = tf.keras.metrics.AUC(curve='PR', name='train_PR_AUC', dtype=tf.float32)

        if validation:
            val_ROC_AUC = tf.keras.metrics.AUC(curve = 'ROC', name='val_ROC_AUC', dtype=tf.float32)
            val_PR_AUC = tf.keras.metrics.AUC(curve = 'PR', name='val_PR_AUC', dtype=tf.float32)

            # TODO: val loss?
            # val_loss = tf.keras.metrics.Mean(name='val_loss', dtype=tf.float32)
        
        print('Setting Up Tensorboard')
        #in case of keyboard interrupt during previous training
        tf.summary.trace_off()
        
        # setting up summary writers
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')

        train_log_dir = log_dir + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        
        if analyse_trace:
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
                loss = loss_obj(label_batch, logits)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            train_ROC_AUC.update_state(label_batch, logits)
            train_PR_AUC.update_state(label_batch, logits)
            return loss


        def val_step(entry):
            audio_batch, label_batch = entry['audio'], entry['tags']
            logits = model(audio_batch)
            # TODO: record loss for val dataset?
            # loss_value = loss(label_batch, logits)
            # val_loss.update_state(loss_value)
            val_ROC_AUC.update_state(label_batch, logits)
            val_PR_AUC.update_state(label_batch, logits)

        @tf.function 
        def distributed_train_body(entry):
            per_replica_losses = strategy.experimental_run_v2(train_step, args=(entry,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        @tf.function
        def distributed_val_body(entry):
            tf.keras.backend.set_learning_phase(0)

            strategy.experimental_run_v2(val_step, args=(entry,))

            tf.keras.backend.set_learning_phase(1)


        # setting up checkpoints
        print('Setting Up Checkpoints')
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

            total_loss = 0.0
            num_batches = 0
            for entry in train_dist_dataset:
                loss = distributed_train_body(entry)            
                total_loss += loss
                num_batches += 1

                if tf.equal(num_batches % 10, 0):
                    tf.print('Epoch',  epoch,'; Step', num_batches, '; loss', loss, '; ROC_AUC', train_ROC_AUC.result(), ';PR_AUC', train_PR_AUC.result())
                    
                    with train_summary_writer.as_default():
                        tf.summary.scalar('ROC_AUC', train_ROC_AUC.result(), step=optimizer.iterations)
                        tf.summary.scalar('PR_AUC', train_PR_AUC.result(), step=optimizer.iterations)
                        tf.summary.scalar('Loss', train_PR_AUC.result(), step=optimizer.iterations)
                        train_summary_writer.flush()

            train_loss = total_loss / num_batches
            # print progress
            tf.print('Epoch', epoch,  ': loss', train_loss, '; ROC_AUC', train_ROC_AUC.result(), '; PR_AUC', train_PR_AUC.result())


            #print progress
            tf.print('Epoch {} --training done'.format(epoch))

            # tensorboard export profiling and record train AUC and loss
            if analyse_trace:
                with prof_summary_writer.as_default():   
                    tf.summary.trace_export(name="trace", 
                                            step=epoch, 
                                            profiler_outdir=os.path.normpath(prof_log_dir)) 

            train_ROC_AUC.reset_states()
            train_PR_AUC.reset_states()

            if validation:
                tf.print('Validation')
                num_batches = 0
                for entry in val_dist_dataset:
                    distributed_val_body(entry) 
                    num_batches += 1

                with val_summary_writer.as_default():
                    tf.summary.scalar('ROC_AUC', val_ROC_AUC.result(), step=epoch)
                    tf.summary.scalar('PR_AUC', val_PR_AUC.result(), step=epoch)
                    val_summary_writer.flush()

                tf.print('Val- Epoch', epoch, ': ROC_AUC', val_ROC_AUC.result(), '; PR_AUC', val_PR_AUC.result())
                
                # reset val metric per epoch
                val_ROC_AUC.reset_states()
                val_PR_AUC.reset_states()

            checkpoint_path = os.path.join(model_dir, 'epoch_{}.ckpt'.format(epoch))
            saved_path = checkpoint.save(checkpoint_path)
            tf.print('Saving model as TF checkpoint: {}'.format(saved_path))

            #report time
            time_taken = time.time()-start_time
            tf.print('Time taken for epoch {}: {}s'.format(epoch, time_taken))

def main(tfrecord_dir, frontend_mode, config_dir, train_val_test_split=(70, 10, 20),
         batch_size=32, validation=True, shuffle=True, buffer_size=10000, 
         window_length=15, random=False, with_tags=None, merge_tags=None,
         log_dir = 'logs/trial1/', with_tids=None, num_epochs=5):
   
    '''Combines data input pipeline, networks, train and validation loops to 
        perform model training.

    Parameters
    ----------
        
    frontend_mode: str
        'waveform' or 'log-mel-spectrogram', indicating the format of the
        audio inputs contained in the tfrecord files.
        
    config_dir: str
        The directory (config.json) or path of where the json file (contains 
        training and dataset configuration info) created in projectname.py 
        is stored.

    train_val_test_split: tuple (a tuple of three integers)
        Specifies the train/validation/test percentage to use when selecting 
        the .tfrecord files.

    batch_size: int
        Specifies the dataset batch_size.

    validation: bool
        If True, validation is performed within each epoch.
        
    shuffle: bool
        If True, shuffles the dataset with buffer size = buffer_size.

    buffer_size: int
        If shuffle is True, sets the shuffle buffer size.

    window_length: list, int
        Specifies the desired window length (in seconds) for the various datasets.

    random: bool
        Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).

    with_tags: list
        If not None, contains the tags to use.

    merge_tags: list
        If not None, contains the lists of tags to be merged together (only applies if with_tags is specified).
    '''
    
    #initialise configuration
    if not os.path.isfile(config_dir):
        config_dir = os.path.join(os.path.normpath(config_dir), 'config.json')
        
    with open(config_dir) as f:
        file = json.load(f)
        
    num_output_neurons = file['dataset_specs']['n_tags']
    y_input = file['dataset_specs']['n_mels']
    lr = file['training_options']['lr']
    num_units = file['training_options']['n_dense_units']
    num_filt = file['training_options']['n_filters']

    strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1'])
    
    print('Preparing Dataset')
    train_dataset, val_dataset = \
    train_cpu.generate_datasets(tfrecord_dir=tfrecord_dir, audio_format=frontend_mode, 
                      train_val_test_split=train_val_test_split, 
                      which = [True, True, False],
                      batch_size=batch_size, shuffle=shuffle, 
                      buffer_size=buffer_size, window_length=window_length, 
                      random=random, with_tags=with_tags, merge_tags=merge_tags,
                      with_tids=with_tids, num_epochs=1)

    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)
    
    print('Train Begin')
    train(frontend_mode=frontend_mode, train_dist_dataset=train_dist_dataset, 
          strategy=strategy, val_dist_dataset=val_dist_dataset, validation=validation,  
          num_epochs=num_epochs, num_output_neurons=num_output_neurons, 
          y_input=y_input, num_units=num_units, num_filt=num_filt, global_batch_size=batch_size,
          lr=lr, log_dir=log_dir)

if __name__ == '__main__':
    #solve the warning--Could not dlopen library 'libcupti.so.10.0' warning
    #https://github.com/google/seq2seq/issues/336
   #os.environ['LD_LIBRARY_PATH'] = "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64"

    main('/srv/data/urop/tfrecords-log-mel-spectrogram', 'log-mel-spectrogram', '/home/calle/', train_val_test_split=(80, 10, 10), shuffle=True, batch_size=128, buffer_size=1000, num_epochs=5)
