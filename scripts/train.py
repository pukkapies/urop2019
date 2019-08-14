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
import model_keras as Model
from datetime import datetime
import os
import time
import projectName_input
import json


def create_config_txt(config_dir, n_tags=155, n_mels=96, lr=0.001, 
                      n_dense_units=1024, n_filters=32):
    '''Create configuration file for training'''
    
    data_params = {'n_tags':n_tags, 'n_mels':n_mels}
    train_params = {'lr':0.001, 'n_dense_units':1024, 'n_filters':32}
    file = {'data_params':data_params, 'train_params':train_params}
    
    with open(os.path.join(os.path.abspath(config_dir),'config.txt'), 'w') as f:
        json.dump(file, f)

def update_config_txt(config_path, new_filename=None, n_tags=None, n_mels=None, 
                       lr=None, n_dense_units=None, n_filters=None):
    '''Update parameters in configuration file produced by create_config_txt()'''
    
    if not os.path.isfile(config_path):
        config_path = os.path.join(config_path, 'config.txt')
    
    config_path = os.path.normpath(config_path)
    with open(config_path) as f:
        file = json.load(f)
        
    if n_tags is not None:
        file['data_params'].update({'n_tags':n_tags})
    if n_mels is not None:
        file['data_params'].update({'n_mels':n_mels})
    if lr is not None:
        file['train_params'].update({'lr':lr})
    if n_dense_units is not None:
        file['train_params'].update({'n_dense_units':n_dense_units})
    if n_filters is not None:
        file['train_params'].update({'n_filters':n_filters})
    
    if new_filename is not None:
        config_path = os.path.join(config_path, new_filename) #replace filename
    
    with open(config_path, 'w') as f:
        json.dump(file, f)

@tf.function
def train_comp(model, optimizer, x_batch_train, y_batch_train, loss):
    '''Optimisation and update gradient'''

    with tf.GradientTape() as tape:
        logits = model(x_batch_train)
                
        loss_value = loss(y_batch_train, logits)
        
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    
    return loss_value, logits    
    
@tf.function
def train_body(dataset, model, optimizer, loss, train_AUC):
    '''Train and update metrics'''
    
         #https://www.tensorflow.org/tensorboard/r2/get_started
    loss_value=0.

    for step, entry in dataset.enumerate():
        x_batch_train, y_batch_train = entry['audio'], entry['tags']
                
        loss_value, logits = train_comp(model, optimizer, x_batch_train, y_batch_train, loss)
            
        train_AUC(y_batch_train, logits)
        
        if tf.equal(step%100, 0):   #change here for printing frequency
            tf.print('Step', step, ': loss', loss_value, '; AUC', train_AUC.result())
            
            
    return loss_value
            
@tf.function      
def val_body(dataset, model, val_AUC):
    '''Validation and update metrics'''
    
    #set training phase =0 to use Dropout and BatchNormalization in test mode
    tf.keras.backend.set_learning_phase(0)
    
    for entry in dataset:
        x_batch_val, y_batch_val = entry['audio'], entry['tags']
            
        val_logits = model(x_batch_val)
        #calculate AUC
        val_AUC(y_batch_val, val_logits)
    
    #set training phase =1
    tf.keras.backend.set_learning_phase(1)

def train(frontend_mode, train_datasets, val_datasets=None, validation=True, 
          num_epochs=10, numOutputNeurons=155, y_input=96, num_units=1024, 
          num_filt=32, lr=0.001, log_dir = 'logs/trial1/'):
    
    #in case of keyboard interrupt during previous training
    tf.summary.trace_off()
    
    #initiate tensorboard
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    train_log_dir = log_dir + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    
    prof_log_dir = log_dir + current_time + '/prof'
    prof_summary_writer = tf.summary.create_file_writer(prof_log_dir)
    
    if validation:
        val_log_dir = log_dir + current_time + '/val'
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    
    
    #import model
    model = Model.build_model(frontend_mode=frontend_mode,
                              numOutputNeurons=numOutputNeurons,
                              y_input=y_input, num_units=num_units, 
                              num_filt=num_filt)
        
    #initialise loss, optimizer, metric
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
    
    # Setting up checkpoints, and loading one if one already exists
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    
    #epoch loop
    for epoch in range(num_epochs):
        start_time = time.time()
        tf.print('Epoch {}'.format(epoch))
        
        #initialise metrics per epoch
        train_AUC = tf.keras.metrics.AUC(name='train_AUC', dtype=tf.float32)
        
        
        tf.summary.trace_on(graph=False, profiler=True)
        #train all batches once
        for idx, dataset in enumerate(train_datasets):
            tf.print('tfrecord {}'.format(idx))
            
            loss_value = train_body(dataset, model, optimizer, loss, train_AUC)
            
            #print progress
            tf.print('tfrecord {} done'.format(idx))
            tf.print('Epoch', epoch, ': tfrecord', idx, '; loss', loss_value, '; AUC', train_AUC.result())
            tf.print(optimizer.iterations)
            tf.print(train_AUC.result())

            # saving checkpoint
            ckpt.step.assign_add(1)
            if int(ckpt.step) % 10 == 0:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

            #log to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('AUC', train_AUC.result(), step=int(optimizer.iterations.numpy()))
                tf.summary.scalar('Loss', loss_value, step=int(optimizer.iterations.numpy()))
            
        #print progress
        tf.print('Epoch {} --training done'.format(epoch))
        tf.print('Epoch', epoch, ': loss', loss_value, '; AUC', train_AUC.result())
        
        #tensorboard export profiling and record train AUC and loss
       
        with prof_summary_writer.as_default():   
            tf.summary.trace_export(
                    name="trace", 
                    step=epoch, profiler_outdir=os.path.normpath(prof_log_dir)) 
            
        #reset train metric per epoch
        train_AUC.reset_states()
        
        #validation
        if validation:
            val_AUC = tf.keras.metrics.AUC(name='val_AUC', dtype=tf.float32)
            #run validation over all batches
            for dataset in val_datasets:
                val_body(dataset, model, val_AUC)
        
            with val_summary_writer.as_default():
                tf.summary.scalar('AUC', val_AUC.result(), step=epoch)
            tf.print('Val- Epoch', epoch, ': AUC', val_AUC.result())
        
            #reset val metric per epoch
            val_AUC.reset_states()  
            
        #report time
        time_taken = time.time()-start_time
        tf.print('Time taken for epoch {}: {}s'.format(epoch, time_taken))

def generate_datasets(tfrecord_dir, audio_format, 
                      train_val_test_split=(70, 10, 20), 
                      batch_size=32, shuffle=True, buffer_size=10000, 
                      window_length=15, random=False, with_tags=None, 
                      with_tids=None, num_epochs=None):
    
    tfrecords = []

    for file in os.listdir(tfrecord_dir):

        if file.endswith(".tfrecord") and file.split('_')[0] == audio_format:

            tfrecords.append(os.path.abspath(os.path.join(tfrecord_dir, file)))
    
    np.random.shuffle(tfrecords)
    
    if isinstance(window_length, int):
        window_length = [window_length]*3
    
    if sum(train_val_test_split) > len(tfrecords):
        raise ValueError('not enough tfrecord files in the directory provided')

    split = [0, train_val_test_split[0], train_val_test_split[0]+train_val_test_split[1],
             sum(train_val_test_split)]
    
    dataset_list = [None, None, None]
    
    
    for num in range(3):
        if train_val_test_split[num] >0:
            dataset_list[num] = projectName_input.genrate_dataset(
                    tfrecords = tfrecords[split[num]:split[num+1]], 
                    audio_format = audio_format, 
                    batch_size=batch_size, 
                    shuffle=shuffle, 
                    buffer_size=buffer_size, 
                    window_length=window_length[num], 
                    random=random, 
                    with_tags=with_tags, 
                    with_tids=with_tids, 
                    num_epochs=num_epochs)
                    
    return dataset_list[0], dataset_list[1], dataset_list[2]

def main(tfrecord_dir, frontend_mode, config_dir, train_val_test_split=(70, 10, 20),
         batch_size=32, validation=True, shuffle=True, buffer_size=10000, 
         window_length=15, random=False, with_tags=None,
         log_dir = 'logs/trial1/', with_tids=None, num_epochs=None):
    
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
    

    train_datasets, val_datasets, test_datasets = \
    generate_datasets(tfrecord_dir=tfrecord_dir, audio_format=frontend_mode, 
                      train_val_test_split=train_val_test_split, 
                      batch_size=batch_size, shuffle=shuffle, 
                      buffer_size=buffer_size, window_length=window_length, 
                      random=random, with_tags=with_tags, with_tids=with_tids, 
                      num_epochs=num_epochs)
    
    train(frontend_mode=frontend_mode, train_datasets=train_datasets, 
          val_datasets=val_datasets, validation=validation,  
          num_epochs=num_epochs, numOutputNeurons=numOutputNeurons, 
          y_input=y_input, num_units=num_units, num_filt=num_filt, 
          lr=lr, log_dir = log_dir)


    
    
    
    
    
    
    
    
