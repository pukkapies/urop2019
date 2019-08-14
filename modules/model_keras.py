''' Contains skeleton model for training 

Notes
-----
This module reproduces the skeleton model proposed by (Pons, et al., 2018) in
tensorflow 2.0 keras syntax. For more information, please refer to 
(Pons, et al., 2018).

This module can be divded into three parts:
    1. Model frontends for waveform and log-mel-spectrogram input respectively.
    2. Model backend for both model frontends
    3. A final model combining a frontend and the backend.
    
Functions
---------
- wave_frontend
    Model frontend for waveform input.

- spec_frontend
    Model frontend for log-mel-spectrogram input.

- backend
    Model backend for waveform and log-mel-spectrogram input.

- build_model
    Generate a final model by combining a frontend with the backend.
    
Reference
---------
Pons, J. et al., 2018. END-TO-END LEARNING FOR MUSIC AUDIO TAGGING AT SCALE. Paris, s.n., pp. 637-644.


'''
"""

Copyright 2017-2019 Pandora Media, Inc.

Redistribution and use in source and binary forms, with or without

modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,

this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,

this list of conditions and the following disclaimer in the documentation

and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors

may be used to endorse or promote products derived from this software without

specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"

AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE

IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE

ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE

LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR

CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF

SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS

INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN

CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)

ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE

POSSIBILITY OF SUCH DAMAGE.

"""
import tensorflow as tf
from tensorflow.keras.layers import Add, AveragePooling2D, BatchNormalization, \
Concatenate, Conv1D, Conv2D, Dense, Dropout, Flatten, Lambda, MaxPool1D, \
MaxPool2D, Permute, ZeroPadding2D
        
    
    
def wave_frontend(Input):
    '''Model frontend for waveform input.'''
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


def spec_frontend(Input, y_input=96, num_filt=32):
    '''Model frontend for log-mel-spectrogram input, see documentation on 
    build_model() for more details.'''
    
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


def backend(Input, numOutputNeurons, num_units=1024):
    '''Model backend for waveform and log-mel-spectrogram input, see 
    documentation on build_model() for more detail.'''
    
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
    
    return Dense(activation='sigmoid', units=numOutputNeurons,
                 kernel_initializer=initializer, name='dense2_back')(dense_dropout)
        

def build_model(frontend_mode, numOutputNeurons=155, 
                y_input=96, num_units=1024,
                num_filt=32):
    '''Generate a final model by combining a frontend with the backend.
    
    Parameters
    ----------
    frontend_mode: string
        'waveform', or 'log-mel-spectrogram' to indicate the frontend model.
        
    numOutputNeurons: int
        The dimension of the prediction array for each audio input. This should
        be set to the length of the a one-hot encoding of tags.
        
    y_input: int or None
        For waveform frontend, y_input will not affect the output of function.
        For log-mel-spectrogram frontend, this is the height of the spectrogram
        and should therefore be set as the number of mel bands of the 
        spectrogram.
        
    num_units: int
        The number of neurons in the dense hidden layer of the backend.
        
    num_filts: int
        For waveform, num_filts will not affect the ouput of function. For 
        log-mel-spectrogram, this is the number of filters of the first CNN
        layer. See (Pons, et al., 2018) for more details.
    
    '''

    if frontend_mode == 'waveform':
        Input = tf.keras.Input(shape=[None])
        front_out = wave_frontend(Input)
        model = tf.keras.Model(Input,
                               backend(front_out,
                                       numOutputNeurons=numOutputNeurons,
                                       num_units=num_units))
        return model
    
    elif frontend_mode == 'log-mel-spectrogram':
        Input = tf.keras.Input(shape=[y_input, None])
        front_out = spec_frontend(Input, y_input=y_input, num_filt=num_filt)
        model = tf.keras.Model(Input,
                               backend(front_out,
                                       numOutputNeurons=numOutputNeurons,
                                       num_units=num_units))
        return model
    
    else: 
        print('Please specify the frontend_mode: "waveform" or "log-mel-spectrogram"')
    


        