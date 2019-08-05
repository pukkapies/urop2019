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
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, \
Conv1D, Conv2D, Dense, Dropout, Flatten, MaxPool1D, MaxPool2D

def wave_frontend(Input, is_training=True):
    initializer = tf.keras.initializers.VarianceScaling()
    
    Input = tf.expand_dims(Input, 2)
    
    #conv0
    conv0 = Conv1D(filters=64, kernel_size=3, strides=3, padding='valid',
                   activation='relu', kernel_initializer=initializer)(Input)
    bn_conv0 = BatchNormalization()(conv0, training=is_training)
            
    #conv1
    conv1 = Conv1D(filters=64, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer)(bn_conv0)
    bn_conv1 = BatchNormalization()(conv1, training=is_training)
    pool1 = MaxPool1D(pool_size=3, strides=3)(bn_conv1)
            
    #conv2
    conv2 = Conv1D(filters=64, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer)(pool1)
    bn_conv2 = BatchNormalization()(conv2, training=is_training)
    pool2 = MaxPool1D(pool_size=3, strides=3)(bn_conv2)
            
    #conv3
    conv3 = Conv1D(filters=128, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer)(pool2) 
    bn_conv3 = BatchNormalization()(conv3, training=is_training)
    pool3 = MaxPool1D(pool_size=3, strides=3)(bn_conv3)
            
    #conv4
    conv4 = Conv1D(filters=128, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer)(pool3) 
    bn_conv4 = BatchNormalization()(conv4, training=is_training)
    pool4 = MaxPool1D(pool_size=3, strides=3)(bn_conv4)
            
    #conv5
    conv5 = Conv1D(filters=128, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer)(pool4)
    bn_conv5 = BatchNormalization()(conv5, training=is_training)
    pool5 = MaxPool1D(pool_size=3, strides=3)(bn_conv5)
            
    #conv6
    conv6 = Conv1D(filters=256, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer)(pool5)
    bn_conv6 = BatchNormalization()(conv6, training=is_training)
    pool6 = MaxPool1D(pool_size=3, strides=3)(bn_conv6)
    
    return tf.expand_dims(pool6, [3])


def spec_frontend(Input, y_input, is_training=True, num_filt=32):
    initializer = tf.keras.initializers.VarianceScaling()
    
    Input = tf.expand_dims(Input, 3)
    
    #padding for time axis
    Input_pad_7 = tf.pad(Input, [[0, 0], [0, 0], [3, 3], [0, 0]],
                         'CONSTANT')
    Input_pad_3 = tf.pad(Input, [[0, 0], [0, 0], [1, 1], [0, 0]],
                         'CONSTANT')
    
    #conv1
    conv1 = Conv2D(filters=num_filt, 
               kernel_size=[int(0.9 * y_input), 7],
               padding='valid', activation='relu',
               kernel_initializer=initializer)(Input_pad_7)
    bn_conv1 = BatchNormalization()(conv1, training=is_training)
    pool1 = MaxPool2D(pool_size=[conv1.shape[1], 1], 
                      strides=[conv1.shape[1], 1])(bn_conv1)
    p1 = tf.squeeze(pool1, [1])
    
    #conv2
    conv2 = Conv2D(filters=num_filt*2,
               kernel_size=[int(0.9 * y_input), 3],
               padding='valid', activation='relu',
               kernel_initializer=initializer)(Input_pad_3)
    bn_conv2 = BatchNormalization()(conv2, training=is_training)
    pool2 = MaxPool2D(pool_size=[conv2.shape[1], 1], 
                      strides=[conv2.shape[1], 1])(bn_conv2)
    p2 = tf.squeeze(pool2, [1])
    
    #conv3
    conv3 = Conv2D(filters=num_filt*4,
               kernel_size=[int(0.9 * y_input), 1], 
               padding='valid', activation='relu',
               kernel_initializer=initializer)(Input)
    bn_conv3 = BatchNormalization()(conv3, training=is_training)
    pool3 = MaxPool2D(pool_size=[conv3.shape[1], 1], 
                      strides=[conv3.shape[1], 1])(bn_conv3)
    p3 = tf.squeeze(pool3, [1])
    
    #conv4
    conv4 = Conv2D(filters=num_filt,
               kernel_size=[int(0.4 * y_input), 7],
               padding='valid', activation='relu',
               kernel_initializer=initializer)(Input_pad_7)
    bn_conv4 = BatchNormalization()(conv4, training=is_training)
    pool4 = MaxPool2D(pool_size=[conv4.shape[1], 1], 
                  strides=[conv4.shape[1], 1])(bn_conv4)
    p4 = tf.squeeze(pool4, [1])
    
    #conv5
    conv5 = Conv2D(filters=num_filt*2,
               kernel_size=[int(0.4 * y_input), 3],
               padding='valid', activation='relu',
               kernel_initializer=initializer)(Input_pad_3)
    bn_conv5 = BatchNormalization()(conv5, training=is_training)
    pool5 = MaxPool2D(pool_size=[conv5.shape[1], 1], 
                      strides=[conv5.shape[1], 1])(bn_conv5)
    p5 = tf.squeeze(pool5, [1])
    
    #conv6
    conv6 = Conv2D(filters=num_filt*4,
               kernel_size=[int(0.4 * y_input), 1],
               padding='valid', activation='relu',
               kernel_initializer=initializer)(Input)
    bn_conv6 = BatchNormalization()(conv6, training=is_training)
    pool6 = MaxPool2D(pool_size=[conv6.shape[1], 1], 
                  strides=[conv6.shape[1], 1])(bn_conv6)
    p6 = tf.squeeze(pool6, [1])
    
    
    #average pooling
    avg_pool = AveragePooling2D(pool_size=[y_input, 1], 
                             strides=[y_input, 1])(Input)
    avg_pool = tf.squeeze(avg_pool, [1])
    
    
    #conv7
    conv7 = Conv1D(filters=num_filt, kernel_size=165,
                   padding='same', activation='relu',
                   kernel_initializer=initializer)(avg_pool)
    bn_conv7 = BatchNormalization()(conv7, training=is_training)
    
    #conv8
    conv8 = Conv1D(filters=num_filt*2, kernel_size=128,
                   padding='same', activation='relu',
                   kernel_initializer=initializer)(avg_pool)
    bn_conv8 = BatchNormalization()(conv8, training=is_training)
    
    #conv9
    conv9 = Conv1D(filters=num_filt*4, kernel_size=64,
                   padding='same', activation='relu',
                   kernel_initializer=initializer)(avg_pool)
    bn_conv9 = BatchNormalization()(conv9, training=is_training)
    
    #conv10
    conv10 = Conv1D(filters=num_filt*8, kernel_size=32,
                   padding='same', activation='relu',
                   kernel_initializer=initializer)(avg_pool)
    bn_conv10 = BatchNormalization()(conv10, training=is_training)
    
    concat = tf.concat([p1, p2, p3, p4, p5, p6, bn_conv7, bn_conv8,
                        bn_conv9, bn_conv10], 2)
    
    return tf.expand_dims(concat, 3)


def backend(Input, numOutputNeurons, is_training=True, num_units=1024):
    initializer = tf.keras.initializers.VarianceScaling()
    
    #conv1
    conv1 = Conv2D(filters=512, kernel_size=[7, Input.shape[2]],
                   padding='valid', activation='relu',
                   kernel_initializer=initializer)(Input)
    bn_conv1 = BatchNormalization()(conv1, training=is_training)
    bn_conv1_t = tf.transpose(bn_conv1, [0, 1, 3, 2])
    
    #conv2, residue connection
    bn_conv1_pad = tf.pad(bn_conv1_t, [[0, 0], [3, 3], [0, 0], [0, 0]],
                          'CONSTANT')
    conv2 = Conv2D(filters=512, kernel_size=[7, bn_conv1_pad.shape[2]],
                   padding='valid', activation='relu',
                   kernel_initializer=initializer)(bn_conv1_pad)
    conv2_t = tf.transpose(conv2, [0, 1, 3, 2])
    bn_conv2 = BatchNormalization()(conv2_t, training=is_training)
    res_conv2 = tf.math.add(bn_conv2, bn_conv1_t)
    
    
    #temporal pooling
    pool1 = MaxPool2D(pool_size=[2, 1], strides=[2, 1])(res_conv2)
    
    #conv3, residue connection
    pool1_pad = tf.pad(pool1, [[0, 0], [3, 3], [0, 0], [0, 0]],
                       'CONSTANT')
    conv3 = Conv2D(filters=512, kernel_size=[7, pool1_pad.shape[2]],
                   padding='valid', activation='relu',
                   kernel_initializer=initializer)(pool1_pad)
    conv3_t = tf.transpose(conv3, [0, 1, 3, 2])
    bn_conv3 = BatchNormalization()(conv3_t, training=is_training)
    res_conv3 = tf.math.add(bn_conv3, pool1)
    
    #global pooling
    max_pool2 = tf.reduce_max(res_conv3, axis=1)
    avg_pool2, var_pool2 = tf.nn.moments(res_conv3, axes=[1])
    pool2 = tf.concat([max_pool2, avg_pool2], 2)
    flat_pool2 = Flatten()(pool2)
    
    #dense1
    flat_pool2_dropout = Dropout(rate=0.5)(flat_pool2, training=is_training)
    dense = Dense(units=num_units, activation='relu',
                  kernel_initializer=initializer)(flat_pool2_dropout)
    bn_dense = BatchNormalization()(dense, training=is_training)
    dense_dropout = Dropout(rate=0.5)(bn_dense, training=is_training)
    
    return Dense(activation='sigmoid', units=numOutputNeurons,
                 kernel_initializer=initializer)(dense_dropout)
    
    
def build_model(frontend_mode, numOutputNeurons, 
                y_input=None, is_training=True, num_units=1024,
                num_filt=32):

    if frontend_mode == 'wave':
        Input = tf.keras.Input(shape=[None])
        front_out = wave_frontend(Input)
        model = tf.keras.Model(Input,
                               backend(front_out,
                                       numOutputNeurons=numOutputNeurons,
                                       num_units=num_units))
        return model
    
    elif frontend_mode == 'spec':
        Input = tf.keras.Input(shape=[y_input, None])
        front_out = spec_frontend(Input, y_input=y_input, num_filt=num_filt)
        model = tf.keras.Model(Input,
                               backend(front_out,
                                       numOutputNeurons=numOutputNeurons,
                                       num_units=num_units))
        return model
    
    else: 
        print('Please specify the frontend_mode: "wave" or "spec"')
    
        