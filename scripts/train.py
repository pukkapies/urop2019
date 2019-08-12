'''
TODO LIST:
- import test.py
- solve incompatible dimension between model_keras.py and test.py
- add printing
- fix tensorboard and make it run on Boden (create graph, profile)
- create json file
- evaluation method (include test phase and convert tag_num to tag)

'''

import sys
sys.path.insert(0, 'C://Users/hcw10/UROP2019')


import tensorflow as tf
import model_keras as Model
from datetime import datetime
import os
import time


def train_exp(frontend_mode, numOutputNeurons, train_datasets, val_datasets,
          y_input=None, is_training=True, num_units=1024,
          num_filt=32, n_epoch=10, lr=0.001):
    
    #in case of keyboard interrupt during previous training
    #tf.summary.trace_off()
    #initiate tensorboard
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    train_log_dir = 'logs/gradient_tape2/' + current_time + '/train'
    val_log_dir = 'logs/gradient_tape2/' + current_time + '/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    
    tf.summary.trace_on(graph=True, profiler=False)
    
    #import model
    model = Model.build_model(frontend_mode=frontend_mode,
                              numOutputNeurons=numOutputNeurons,
                              y_input=y_input, is_training=is_training,
                              num_units=num_units, num_filt=num_filt)
    
    
    #export tensorboard graph
    with train_summary_writer.as_default():
        tf.summary.trace_export(
                name="my_func_trace",
                step=0)
        
    #initialise loss, optimiser, metric
    loss = tf.keras.losses.MeanSquaredError()
    optimiser = tf.keras.optimizers.Nadam(learning_rate=lr)
    
    #epoch loop
    for epoch in range(n_epoch):
        tf.print('Epoch {}'.format(epoch))
        
        #initialise metrics per epoch
        train_AUC = tf.keras.metrics.AUC(name='train_AUC', dtype=tf.float32)
        val_AUC = tf.keras.metrics.AUC(name='val_AUC', dtype=tf.float32)
        
        tf.summary.trace_on(graph=False, profiler=True)
        #train all batches once
        for idx, dataset in enumerate(train_datasets):
            
            loss_value = train_body(dataset, model, optimiser, loss, train_AUC)
            print(loss_value)
        
        #tensorboard export profiling and record train AUC
        with train_summary_writer.as_default(): #
            tf.summary.trace_export(
                    name="my_func_trace", 
                    step=0, profiler_outdir=os.path.normpath(train_log_dir)) #
        
        with train_summary_writer.as_default():
            tf.summary.scalar('AUC', train_AUC.result(), step=epoch)
            
        #reset train metric per epoch
        train_AUC.reset_states()
        
        #run validation over all batches
        for dataset in val_datasets:
            val_body(dataset, model, val_AUC)
        
        with val_summary_writer.as_default():
            tf.summary.scalar('AUC', val_AUC.result(), step=epoch)
        
        #reset val metric per epoch
        val_AUC.reset_states()  


@tf.function
def train_comp(model, optimiser, x_batch_train, y_batch_train, loss):
    #https://www.tensorflow.org/beta/guide/autograph#define_the_training_loop
    with tf.GradientTape() as tape:
        logits = model(x_batch_train)
                
        loss_value = loss(y_batch_train, logits)
        
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimiser.apply_gradients(zip(grads, model.trainable_weights))
    
    
    return loss_value, logits


def transformation(x_batch_train, y_batch_train):

    x_batch_train = tf.sparse.to_dense(x_batch_train)
    x_batch_train = tf.expand_dims(x_batch_train, 0)
    y_batch_train = tf.reshape(y_batch_train, (-1, 155))
    return x_batch_train, y_batch_train
    
    
@tf.function
def train_body(dataset, model, optimiser, loss, train_AUC):
    
         #https://www.tensorflow.org/tensorboard/r2/get_started
    loss_value=0.
        
    for step, entry in dataset.enumerate():
        x_batch_train, y_batch_train, tid = entry['audio'], entry['tags'], entry['tid']
        x_batch_train, y_batch_train = transformation(x_batch_train, y_batch_train)
                ##!!!currently incompatible with testing.py
                
        loss_value, logits = train_comp(model, optimiser, x_batch_train, y_batch_train, loss)
            
        train_AUC(y_batch_train, logits)
            
    return loss_value
            
@tf.function      
def val_body(dataset, model, val_AUC):

    #set training phase =0 to use Dropout and BatchNormalization in test mode
    tf.keras.backend.set_learning_phase(0)
    
    for entry in dataset:
        x_batch_val, y_batch_val, tid = entry['audio'], entry['tags'], entry['tid']
        x_batch_val, y_batch_val = transformation(x_batch_val, y_batch_val)
                ##!!!currently incompatible with testing.py
            
        val_logits = model(x_batch_val)
        #calculate AUC
        val_AUC(y_batch_val, val_logits)
    
    #set training phase =1
    tf.keras.backend.set_learning_phase(1)
            
       

#####################################################################################


def train(frontend_mode, numOutputNeurons, train_datasets, val_datasets,
          y_input=None, is_training=True, num_units=1024,
          num_filt=32, n_epoch=10, lr=0.001):
    
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
 
    tf.summary.trace_on(graph=True, profiler=False)
    model = Model.build_model(frontend_mode=frontend_mode,
                              numOutputNeurons=numOutputNeurons,
                              y_input=y_input, is_training=is_training,
                              num_units=num_units, num_filt=num_filt)
    
    with train_summary_writer.as_default():
        tf.summary.trace_export(
                name="my_func_trace",
                step=0)
    
    loss = tf.keras.losses.MeanSquaredError()
    optimiser = tf.keras.optimizers.Nadam(learning_rate=lr)
    
    
    #delete later:
    x_train, y_train = exp_spec()
    x_train = tf.constant(x_train, dtype=tf.float32)
    y_train = tf.constant(y_train, dtype=tf.float32)
    
    for epoch in range(n_epoch):
        print('Epoch {}'.format(epoch))
        
         #https://www.tensorflow.org/tensorboard/r2/get_started
        train_AUC = tf.keras.metrics.AUC(name='train_AUC', dtype=tf.float32)
        val_AUC = tf.keras.metrics.AUC(name='val_AUC', dtype=tf.float32)
        tf.summary.trace_on(graph=False, profiler=True)##
        for idx, dataset in enumerate(train_datasets):
           
            for step, entry in enumerate(dataset):
                x_batch_train, y_batch_train, tid = entry['audio'], entry['tags'], entry['tid']
                x_batch_train = tf.sparse.to_dense(x_batch_train)
                x_batch_train = tf.expand_dims(x_batch_train, 0)
                y_batch_train = tf.reshape(y_batch_train, (1, 155))
                ##!!!currently incompatible with testing.py
                
                start = time.time()
                with tf.GradientTape() as tape:
                    #tf.summary.trace_on(graph=False, profiler=True)
                    logits = model(x_batch_train)
                    #with train_summary_writer.as_default():
                    #    tf.summary.trace_export(
                    #            name="my_func_trace",
                    #            step=0, profiler_outdir=os.path.normpath(train_log_dir))
                    loss_value = loss(y_batch_train, logits)
        
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimiser.apply_gradients(zip(grads, model.trainable_weights))
        
                train_AUC(y_batch_train, logits)
                duration = time.time()-start
                print(duration)
        with train_summary_writer.as_default(): #
            tf.summary.trace_export(
                    name="my_func_trace", 
                    step=0, profiler_outdir=os.path.normpath(train_log_dir)) #

        
        with train_summary_writer.as_default():
            tf.summary.scalar('AUC', train_AUC.result(), step=epoch)
            
        print(loss_value.numpy())
            
        train_AUC.reset_states()
        
        for val_dataset in val_datasets:
            
            for entry in val_dataset:
                x_batch_val, y_batch_val, tid = entry['audio'], entry['tags'], entry['tid']
                x_batch_val = tf.sparse.to_dense(x_batch_val)
                x_batch_val = tf.expand_dims(x_batch_val, 0)
                y_batch_val = tf.reshape(y_batch_val, (1, 155))
                ##!!!currently incompatible with testing.py
            
                val_logits = model(x_batch_val)
                #calculate AUC
                val_AUC(y_batch_val, val_logits)
            
        with val_summary_writer.as_default():
            tf.summary.scalar('AUC', val_AUC.result(), step=epoch)
            
        val_AUC.reset_states()
        
            
    
    
    
#    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
#    log_dir = 'logs/gradient_tape/' + current_time + '/train'
#    need to write log_dir
#    model = Model.build_model(frontend_mode='spec',
#                              numOutputNeurons=155,
#                              y_input=96, is_training=True,
#                              num_units=1024, num_filt=96)
    
#    model.compile(optimizer='adam',
#                  metrics=['AUC'], 
#                  loss='mean_squared_error')
    
#    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.normpath(log_dir), histogram_freq=1)
    
#    #delete later
#    x_train, y_train = exp_spec()
#    x_train, y_train = tf.constant(x_train, dtype=tf.float32), tf.constant(y_train, dtype=tf.float32)
#    print('start')
#    model.fit(x=x_train, y=y_train, epochs=10, verbose=2, callbacks=[tensorboard_callback])
    


def exp_spec():
    import librosa
    import numpy as np
    file = np.load('C://Users/hcw10/UROP2019/1109.npz')
    array = file['array']
    sr = file['sr']
    array = array[:, :sr*15]
    array = librosa.core.to_mono(array)
    array = librosa.resample(array, sr, 16000)
    array = np.log(librosa.feature.melspectrogram(array, 16000))
    a = [0,0]
    a[0] = array
    a[1] = array
    array = np.array(a)[:, :96, :]
    return array, np.array([[1],[0]])

def exp_wave():
    import librosa
    import numpy as np
    file = np.load('C://Users/hcw10/UROP2019/1109.npz')
    array = file['array']
    sr = file['sr']
    array = array[:, :sr*15]
    array = librosa.core.to_mono(array)
    array = librosa.resample(array, sr, 16000)
    a = [0,0]
    a[0] = array
    a[1] = array
    return a, np.array([[1],[0]])
    
    
    
    
    
    
    
    
    