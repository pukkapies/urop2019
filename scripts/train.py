'''
TODO LIST:
- add printing
- fix tensorboard and make it run on Boden (create graph, profile)
- evaluation method (include test phase and convert tag_num to tag)

'''

import sys
sys.path.insert(0, 'C://Users/hcw10/UROP2019')

import tensorflow as tf
import model as Model
from datetime import datetime
import os
import time
import projectName_input
import json




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
        loss_value = tf.reduce_mean(loss_value)
        train_AUC(y_batch_train, logits)
        
        if tf.equal(step%1, 0):   #change here for printing frequency
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

def train(frontend_mode, train_dataset, val_dataset=None, validation=True, 
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
    loss = tf.nn.sigmoid_cross_entropy_with_logits
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
        loss_value = train_body(train_dataset, model, optimizer, loss, train_AUC)
        
            
            #print progress
        tf.print('Epoch', epoch,  ': loss', loss_value, '; AUC', train_AUC.result())

        # saving checkpoint
        ckpt.step.assign_add(1)
        if int(ckpt.step) % 10 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

            #log to tensorboard
        with train_summary_writer.as_default():
            tf.summary.scalar('AUC', train_AUC.result(), step=epoch)
            tf.summary.scalar('Loss', loss_value, step=epoch)
            
        #print progress
        tf.print('Epoch {} --training done'.format(epoch))
        
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
            val_body(val_dataset, model, val_AUC)
        
            with val_summary_writer.as_default():
                tf.summary.scalar('AUC', val_AUC.result(), step=epoch)
            tf.print('Val- Epoch', epoch, ': AUC', val_AUC.result())
        
            #reset val metric per epoch
            val_AUC.reset_states()  
            
        #report time
        time_taken = time.time()-start_time
        tf.print('Time taken for epoch {}: {}s'.format(epoch, time_taken))
    # saving model
    model.save('/srv/data/urop/model.h5')

def generate_datasets(tfrecord_dir, audio_format, 
                      train_val_test_split=(70, 10, 20),
                      which = None,
                      batch_size=32, shuffle=True, buffer_size=10000, 
                      window_length=15, random=False, with_tags=None, 
                      with_tids=None, num_epochs=None, as_tuple=False):
    ''' Generates.....
    
    Parameters
    ----------
    tfrecord_dir : str
        path to tfrecord directory.

    train_val_test_split : tuple
        contains the number of tfrecord files to be used to generate each of the
        train, val and test datasets.

    which : list of bools
        determines which of the train, val and test datasets to create, useful when
        testing and training in different locations

    batch_size : int

    shuffle : bool
        determines weather to shuffle the entries in each of the datasets or not

    buffer_size : int
        buffer size when shuffling each of the datasets
    
    window_length: int
        length in seconds of window to be extracted from the audio data.

    random : bool
        determines weather to extract window randomly, or from the middle.

    with_tags : list
        tags to be trained on

    with_tids : list
        tids to use

    num_epochs : int or None

    Returns
    -------

    A list of up to three datasets. Contains a train, validation and/or test datasets 
    depending on the which parameter.
    
    ''' 
    tfrecords = []

    for file in os.listdir(tfrecord_dir):

        if file.endswith(".tfrecord") and file.split('_')[0] == audio_format:

            tfrecords.append(os.path.abspath(os.path.join(tfrecord_dir, file)))
    
    if isinstance(window_length, int):
        window_length = [window_length]*3
    
    if sum(train_val_test_split) > len(tfrecords):
        raise ValueError('not enough tfrecord files in the directory provided')

    split = [0, train_val_test_split[0], train_val_test_split[0]+train_val_test_split[1],
             sum(train_val_test_split)]
    
    dataset_list = []
    
    if which is None:
        which = [True if num!=0 else False for num in train_val_test_split]
    
    
    for num, save_bool in enumerate(which):
        if train_val_test_split[num] > 0 and save_bool:
            dataset_list.append(projectName_input.generate_dataset(
                    tfrecords = tfrecords[split[num]:split[num+1]], 
                    audio_format = audio_format, 
                    batch_size=batch_size, 
                    shuffle=shuffle, 
                    buffer_size=buffer_size, 
                    window_length=window_length[num], 
                    random=random, 
                    with_tags=with_tags, 
                    with_tids=with_tids, 
                    num_epochs=num_epochs,
                    as_tuple=as_tuple))
                    
    return dataset_list

def main(tfrecord_dir, frontend_mode, config_dir, train_val_test_split=(70, 10, 20),
         which=None, batch_size=32, validation=True, shuffle=True, buffer_size=10000, 
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
    
    datasets = generate_datasets(tfrecord_dir=tfrecord_dir, audio_format=frontend_mode, 
                                 train_val_test_split=train_val_test_split, 
                                 which = None,
                                 batch_size=batch_size, shuffle=shuffle, 
                                 buffer_size=buffer_size, window_length=window_length, 
                                 random=random, with_tags=with_tags, with_tids=with_tids, 
                                 num_epochs=num_epochs, as_tuple=False)
    
    if validation:
        train_dataset, val_dataset = datasets[0], datasets[1]
    else:
        train_dataset = datasets[0]
        val_dataset = None
    
    train(frontend_mode=frontend_mode, train_dataset=train_dataset, 
          val_dataset=val_dataset, validation=validation,  
          num_epochs=num_epochs, numOutputNeurons=numOutputNeurons, 
          y_input=y_input, num_units=num_units, num_filt=num_filt, 
          lr=lr, log_dir = log_dir)


 
    
    
    
    
    
    
    
