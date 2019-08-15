''' Contains skeleton model for training 


Notes
-----
This module creates a json file to record parameters used in training, and
reproduces the skeleton model proposed by (Pons, et al., 2018) in the
new TensorFlow 2.0 Keras syntax. For more information, please refer to (Pons, et al., 2018).

This module can be divded into four parts:
    1. store configuration;
    2. define model of the frontend for waveform and log-mel-spectrogram input respectively;
    3. define model of the backend for both frontends;
    4. generate final model combining frontend and backend.


Functions
---------
- create_config_json
    Create a txt file storing the parameters.
    
- update_config_json
    Update one or more parameters from a stored txt file.

- wave_frontend
    Model frontend for waveform input.

- log_mel_spec_frontend
    Model frontend for log-mel-spectrogram input.

- backend
    Model backend for both waveform and log-mel-spectrogram input.

- build_model
    Generate model by combining frontend and backend.


Copyright
---------
Copyright 2017-2019 Pandora Media, Inc.

Redistribution and use in source and binary forms, with or without modification, are permitted 
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of 
conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of 
conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or 
promote products derived from this software without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR 
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY 
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


References
----------
    Pons, J. et al., 2018. END-TO-END LEARNING FOR MUSIC AUDIO TAGGING AT SCALE. Paris, s.n., pp. 637-644.
'''

import tensorflow as tf
from tensorflow.keras.layers import Add, AveragePooling2D, BatchNormalization, \
Concatenate, Conv1D, Conv2D, Dense, Dropout, Flatten, Lambda, MaxPool1D, \
MaxPool2D, Permute, ZeroPadding2D

import json
import os
        
def create_config_json(config_dir, n_tags=155, n_mels=96, sample_rate=16000, n_dense_units=1024, n_filters=32):
    ''' Creates configuration file with training specs.

    Parameters
    -----------
    config_dir: str
        The directory where the txt file will be stored.
        
    n_tags: int
        The number of tags in the one-hot encoding.
        
    n_mels: int
        The number of mel-bands used to produce the log-mel-spectrogram.
        
    sample_rate: float
        The sample rate used for saving the tracks.
        
    n_dense_units: int
        The number of neurons in the dense hidden layer of the backend.
        
    n_filters: int
        For waveform, num_filts will not affect the ouput of function. 
        For log-mel-spectrogram, this is the number of filters of the first CNN layer. See (Pons, et al., 2018) for details.
        
    Output
    ------
    config.json: json file
        Contains (a large dictionary containing) three dictionaries:
        - 'dataset_specs': contains specs about the dataset; should not be changed unless dataset has been re-generated with different specs;
        - 'train_params_dataset': contains information about how to parse the dataset (e.g. window length, which tags to read);
        - 'train_params' contains information about training parameters (e.g. learning rate).
    '''
    
    dataset_specs = {
        'n_tags': n_tags, 
        'n_mels': n_mels,
        'sample_rate': sample_rate,
    }

    train_params = {
        'lr': 0.001,
        'n_dense_units': n_dense_units,
        'n_filters': n_filters,
    }

    train_params_dataset = {
        'presets': {
            'tags': [
                ['rock', 'pop', 'electronic', 'dance', 'hip-hop', 'jazz', 'metal'],
                ['rock', 'pop', 'electronic', 'dance', 'hip-hop', 'jazz', 'metal', 'male', 'female', 'instrumental'],
            ],
            'merge_tags': [
                None,
                None,
            ],
        },
        'window_length': 15,
        'window_extract_randomly': False,
        'shuffle': True,
        'shuffle_buffer_size': 10000,
    }

    file = {'dataset_specs': dataset_specs, 'train_params': train_params, 'train_params_dataset': train_params_dataset}
    
    with open(os.path.join(os.path.abspath(config_dir),'config.json'), 'w') as f:
        json.dump(file, f, indent=2, sort_keys=True)

def update_config_json(config_path, new_filename=None, n_tags=None, n_mels=None, 
                       lr=None, n_dense_units=None, n_filters=None):
    '''Update parameters in configuration file produced by create_config_txt()
    
    Parameters
    ----------
    config_path: str
        The directory (if config.txt is the filename) or the exact path of 
        where the txt file is produced by create_config_txt().
        
    new_filename: str/None
        The new filename that contains the updated parameters. If None, the 
        changes will overwrite the original input file.
        
    For other parameters, see documentation of create_config_txt().
    
    Outputs
    -------
    Updated txt file with new filename if specified.
        
    '''
    
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
    
def wave_frontend(Input):
    ''' Creates the frontend model for waveform input. '''

    initializer = tf.keras.initializers.VarianceScaling()
    
    Input = Lambda(lambda x: tf.expand_dims(x, 2), name='expdim_1_wave')(Input)

    #conv0
    conv0 = Conv1D(filters=64, kernel_size=3, strides=3, padding='valid',
                   activation='relu', kernel_initializer=initializer, name='conv0_wave')(Input)
    bn_conv0 = BatchNormalization(name='bn0_wave')(conv0)

    #conv1
    conv1 = Conv1D(filters=64, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer,
                   name='conv1_wave')(bn_conv0)
    bn_conv1 = BatchNormalization(name='bn1_wave')(conv1)
    pool1 = MaxPool1D(pool_size=3, strides=3, name='pool1_wave')(bn_conv1)
    
    #conv2
    conv2 = Conv1D(filters=64, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer,
                   name='conv2_wave')(pool1)
    bn_conv2 = BatchNormalization(name='bn2_wave')(conv2)
    pool2 = MaxPool1D(pool_size=3, strides=3, name='pool2_wave')(bn_conv2)

    #conv3
    conv3 = Conv1D(filters=128, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer,
                   name='conv3_wave')(pool2) 
    bn_conv3 = BatchNormalization(name='bn3_wave')(conv3)
    pool3 = MaxPool1D(pool_size=3, strides=3, name='pool3_wave')(bn_conv3)

    #conv4
    conv4 = Conv1D(filters=128, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer,
                   name='conv4_wave')(pool3) 
    bn_conv4 = BatchNormalization(name='bn4_wave')(conv4)
    pool4 = MaxPool1D(pool_size=3, strides=3, name='pool4_wave')(bn_conv4)
            
    #conv5
    conv5 = Conv1D(filters=128, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer,
                   name='conv5_wave')(pool4)
    bn_conv5 = BatchNormalization(name='bn5_wave')(conv5)
    pool5 = MaxPool1D(pool_size=3, strides=3, name='pool5_wave')(bn_conv5)
            
    #conv6
    conv6 = Conv1D(filters=256, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer,
                   name='conv6_wave')(pool5)
    bn_conv6 = BatchNormalization(name='bn6_wave')(conv6)
    pool6 = MaxPool1D(pool_size=3, strides=3, name='pool6_wave')(bn_conv6)
    
    exp_dim = Lambda(lambda x: tf.expand_dims(x, [3]), name='expdim2_wave')(pool6)
    return exp_dim

def log_mel_spec_frontend(Input, y_input=96, num_filt=32):
    ''' Creates the frontend model for log-mel-spectrogram input. '''
    
    initializer = tf.keras.initializers.VarianceScaling()
    Input = tf.expand_dims(Input, 3)
    
    #padding for time axis
    Input_pad_7 = ZeroPadding2D(((0, 0), (3, 3)), name='pad3_spec')(Input)
    Input_pad_3 = ZeroPadding2D(((0, 0), (1, 1)), name='pad7_spec')(Input)
    
    #conv1
    conv1 = Conv2D(filters=num_filt, 
               kernel_size=[int(0.9 * y_input), 7],
               padding='valid', activation='relu',
               kernel_initializer=initializer, name='conv1_spec')(Input_pad_7)    
    bn_conv1 = BatchNormalization(name='bn1_spec')(conv1)
    pool1 = MaxPool2D(pool_size=[conv1.shape[1], 1], 
                      strides=[conv1.shape[1], 1], name='pool1_spec')(bn_conv1)
    p1 = Lambda(lambda x: tf.squeeze(x, 1), name='sque1_spec')(pool1)
    
    #conv2
    conv2 = Conv2D(filters=num_filt*2,
               kernel_size=[int(0.9 * y_input), 3],
               padding='valid', activation='relu',
               kernel_initializer=initializer, name='conv2_spec')(Input_pad_3)
    bn_conv2 = BatchNormalization(name='bn2_spec')(conv2)
    pool2 = MaxPool2D(pool_size=[conv2.shape[1], 1], 
                      strides=[conv2.shape[1], 1], name='pool2_spec')(bn_conv2)
    p2 = Lambda(lambda x: tf.squeeze(x, 1), name='sque2_spec')(pool2)
    
    #conv3
    conv3 = Conv2D(filters=num_filt*4,
               kernel_size=[int(0.9 * y_input), 1], 
               padding='valid', activation='relu',
               kernel_initializer=initializer, name='conv3_spec')(Input)
    bn_conv3 = BatchNormalization(name='bn3_spec')(conv3)
    pool3 = MaxPool2D(pool_size=[conv3.shape[1], 1], 
                      strides=[conv3.shape[1], 1], name='pool3_spec')(bn_conv3)
    p3 = Lambda(lambda x: tf.squeeze(x, 1), name='sque3_spec')(pool3)
    
    #conv4
    conv4 = Conv2D(filters=num_filt,
               kernel_size=[int(0.4 * y_input), 7],
               padding='valid', activation='relu',
               kernel_initializer=initializer, name='conv4_spec')(Input_pad_7)
    bn_conv4 = BatchNormalization(name='bn4_spec')(conv4)
    pool4 = MaxPool2D(pool_size=[conv4.shape[1], 1], 
                  strides=[conv4.shape[1], 1], name='pool4_spec')(bn_conv4)
    p4 = Lambda(lambda x: tf.squeeze(x, 1), name='sque4_spec')(pool4)
    
    #conv5
    conv5 = Conv2D(filters=num_filt*2,
               kernel_size=[int(0.4 * y_input), 3],
               padding='valid', activation='relu',
               kernel_initializer=initializer, name='conv5_spec')(Input_pad_3)
    bn_conv5 = BatchNormalization(name='bn5_spec')(conv5)
    pool5 = MaxPool2D(pool_size=[conv5.shape[1], 1], 
                      strides=[conv5.shape[1], 1], name='pool5_spec')(bn_conv5)
    p5 = Lambda(lambda x: tf.squeeze(x, 1), name='sque5_spec')(pool5)
    
    #conv6
    conv6 = Conv2D(filters=num_filt*4,
               kernel_size=[int(0.4 * y_input), 1],
               padding='valid', activation='relu',
               kernel_initializer=initializer, name='conv6_spec')(Input)
    bn_conv6 = BatchNormalization(name='bn6_spec')(conv6)
    pool6 = MaxPool2D(pool_size=[conv6.shape[1], 1], 
                  strides=[conv6.shape[1], 1], name='pool6_spec')(bn_conv6)
    p6 = Lambda(lambda x: tf.squeeze(x, 1), name='sque6_spec')(pool6)

    
    #average pooling
    avg_pool = AveragePooling2D(pool_size=[y_input, 1], 
                             strides=[y_input, 1], name='avgpool_spec')(Input)
    avg_pool = Lambda(lambda x: tf.squeeze(x, 1), name='sque7_spec')(avg_pool)
    
    
    #conv7
    conv7 = Conv1D(filters=num_filt, kernel_size=165,
                   padding='same', activation='relu',
                   kernel_initializer=initializer, name='conv7_spec')(avg_pool)
    bn_conv7 = BatchNormalization(name='bn7_spec')(conv7)
    
    #conv8
    conv8 = Conv1D(filters=num_filt*2, kernel_size=128,
                   padding='same', activation='relu',
                   kernel_initializer=initializer, name='conv8_spec')(avg_pool)
    bn_conv8 = BatchNormalization(name='bn8_spec')(conv8)
    
    #conv9
    conv9 = Conv1D(filters=num_filt*4, kernel_size=64,
                   padding='same', activation='relu',
                   kernel_initializer=initializer, name='conv9_spec')(avg_pool)
    bn_conv9 = BatchNormalization(name='bn9_spec')(conv9)
    
    #conv10
    conv10 = Conv1D(filters=num_filt*8, kernel_size=32,
                   padding='same', activation='relu',
                   kernel_initializer=initializer, name='conv10_spec')(avg_pool)
    bn_conv10 = BatchNormalization(name='bn10_spec')(conv10)
    
    concat = Concatenate(2, name='concat_spec')([p1, p2, p3, p4, p5, p6, bn_conv7, bn_conv8,
                          bn_conv9, bn_conv10])
    
    exp_dim = Lambda(lambda x: tf.expand_dims(x, 3), name='expdim1_spec')(concat)
    return exp_dim

def backend(Input, num_output_neurons, num_units=1024):
    ''' Creates the backend model. '''
    
    initializer = tf.keras.initializers.VarianceScaling()
    
    #conv1
    conv1 = Conv2D(filters=512, kernel_size=[7, Input.shape[2]],
                   padding='valid', activation='relu',
                   kernel_initializer=initializer, name='conv1_back')(Input)
    bn_conv1 = BatchNormalization(name='bn1_back')(conv1)
    bn_conv1_t = Permute((1, 3, 2), name='perm1_back')(bn_conv1)
    
    #conv2, residue connection
    bn_conv1_pad = ZeroPadding2D(((3, 3), (0, 0)), name='pad3_1_back')(bn_conv1_t)
    conv2 = Conv2D(filters=512, kernel_size=[7, bn_conv1_pad.shape[2]],
                   padding='valid', activation='relu',
                   kernel_initializer=initializer, name='conv2_back')(bn_conv1_pad)
    conv2_t = Permute((1,3,2), name='perm2_back')(conv2)
    bn_conv2 = BatchNormalization(name='bn2_back')(conv2_t)
    res_conv2 = Add(name='add1_back')([bn_conv2, bn_conv1_t])
    
    #temporal pooling
    pool1 = MaxPool2D(pool_size=[2, 1], strides=[2, 1], name='pool1_back')(res_conv2)
    
    #conv3, residue connection
    pool1_pad = ZeroPadding2D(((3, 3), (0, 0)), name='pad3_2_back')(pool1)
    conv3 = Conv2D(filters=512, kernel_size=[7, pool1_pad.shape[2]],
                   padding='valid', activation='relu',
                   kernel_initializer=initializer, name='conv3_back')(pool1_pad)
    conv3_t = Permute((1, 3, 2), name='perm3_back')(conv3)
    bn_conv3 = BatchNormalization(name='bn3_back')(conv3_t)
    res_conv3 = Add(name='add2_back')([bn_conv3, pool1])
    
    #global pooling
    max_pool2 = Lambda(lambda x: tf.keras.backend.max(x, axis=1), name='glo_max_back')(res_conv3)
    avg_pool2, var_pool2 = Lambda(lambda x: tf.nn.moments(x, axes=[1]), name='moment_back')(res_conv3)
    pool2 = Concatenate(2, name='concat_back')([max_pool2, avg_pool2])
    flat_pool2 = Flatten()(pool2)
    
    #dense1
    flat_pool2_dropout = Dropout(rate=0.5, name='drop1_back')(flat_pool2)
    dense = Dense(units=num_units, activation='relu',
                  kernel_initializer=initializer, name='dense1_back')(flat_pool2_dropout)
    bn_dense = BatchNormalization(name='bn_dense_back')(dense)
    dense_dropout = Dropout(rate=0.5, name='drop2_back')(bn_dense)
    
    return Dense(activation='sigmoid', units=num_output_neurons,
                 kernel_initializer=initializer, name='dense2_back')(dense_dropout)

def build_model(frontend_mode, num_output_neurons=155, y_input=96, num_units=1024, num_filt=32):
    ''' Generates the final model by combining frontend and backend.
    
    Parameters
    ----------
    frontend_mode: {'waveform', 'log-mel-spectrogram'} 
        Specifies the frontend model.
        
    num_output_neurons: int
        The dimension of the prediction array for each audio input. This should
        be set to the length of the a one-hot encoding of tags.
        
    y_input: int, None
        For waveform frontend, y_input will not affect the output of the function.
        For log-mel-spectrogram frontend, this is the height of the spectrogram and should therefore be set as the 
        number of mel bands in the spectrogram.
        
    num_units: int
        The number of neurons in the dense hidden layer of the backend.
        
    num_filts: int
        For waveform, num_filts will not affect the ouput of the function. 
        For log-mel-spectrogram, this is the number of filters of the first CNN layer. See (Pons, et al., 2018) for more details.
    '''

    if frontend_mode == 'waveform':
        Input = tf.keras.Input(shape=[None])
        front_out = wave_frontend(Input)
        model = tf.keras.Model(Input,
                               backend(front_out,
                                       num_output_neurons=num_output_neurons,
                                       num_units=num_units))
        return model
    
    elif frontend_mode == 'log-mel-spectrogram':
        Input = tf.keras.Input(shape=[y_input, None])
        front_out = log_mel_spec_frontend(Input, y_input=y_input, num_filt=num_filt)
        model = tf.keras.Model(Input,
                               backend(front_out,
                                       num_output_neurons=num_output_neurons,
                                       num_units=num_units))
        return model
    
    else:
        raise ValueError('please specify the frontend_mode: "waveform" or "log-mel-spectrogram"')
    


        