import tensorflow as tf
import model_keras as Model
from datetime import datetime

#sys.path.insert(0, 'C://Users/hcw10/UROP2019')

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

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
    array = np.array(a)[:100, :96, :]
    return array, np.array([[1],[0]])


def train(frontend_mode, numOutputNeurons, 
          y_input=None, is_training=True, num_units=1024,
          num_filt=32, epoch=10, lr=0.001):
 
    model = Model.build_model(frontend_mode=frontend_mode,
                              numOutputNeurons=numOutputNeurons,
                              y_input=y_input, is_training=is_training,
                              num_units=num_units, num_filt=num_filt)
    
    loss = tf.keras.losses.MeanSquaredError()
    optimiser = tf.keras.optimizers.Nadam(learning_rate=lr)
    
    
    #delete later:
    x_train, y_train = exp_spec()
    x_train = tf.constant(x_train, dtype=tf.float32)
    y_train = tf.constant(y_train, dtype=tf.float32)
    
    for idx in range(epoch):
        print('Epoch {}'.format(idx))
        
        #need to add enumerate dataset 
        #https://www.tensorflow.org/beta/guide/keras/training_and_evaluation#low-level_handling_of_metrics
        
        with tf.GradientTape() as tape:
            logits = model(x_train)
            loss_value = loss(y_train, logits)
            
        grads = tape.gradient(loss_value, model.trainable_weights)
        
        optimiser.apply_gradients(zip(grads, model.trainable_weights))
        
        
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss_value, step=idx)
            
        
        print(loss_value.numpy())
            
    
    
    
    
#    log_dir = 'logs/fit/'+datetime.now().strftime('%Y%m%d-%H%M%S')
#    
#    model = Model.build_model(frontend_mode=frontend_mode,
#                              numOutputNeurons=numOutputNeurons,
#                              y_input=y_input, is_training=is_training,
#                              num_units=num_units, num_filt=num_filt)
#    
#    model.compile(optimizer='adam',
#                  metrics=['AUC'], 
#                  loss='mean_squared_error')
#    
#    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
#                                                          histogram_freq=1)
#    
#    #delete later
#    x_train, y_train = exp_spec()
#    print('start')
#    model.fit(x=x_train, y=y_train, epochs=10, verbose=2, 
#              callbacks=[tensorboard_callback])
    
    
    
    
    
    
    
    
    
    