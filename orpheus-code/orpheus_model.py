''' Contains tools to build the training model and set all the various training parameters.


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
- write_config_json
    Write the .json file storing the training parameters. See inline documentation for more details.

- parse_config_json
    Parse the .json file storing the training parameters.

- frontend_wave
    Model frontend for waveform input.

- frontend_log_mel_spect
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


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS' AND ANY EXPRESS OR 
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

import argparse
import json
import os
import re

from _ctypes import PyObj_FromPtr

import numpy as np
import tensorflow as tf

import lastfm
        
def write_config_json(config_path, **kwargs):
    ''' Write an "empty" configuration file for training specs.

    Parameters
    -----------
    config_path: str
        The path to the .json file.
        
    Outputs
    -------
    config.json: json file
        Contains (a large dictionary containing) three dictionaries:
        - 'dataset_specs': contains specs about the dataset; should not be changed unless dataset has been re-generated with different specs;
        - 'train_options_dataset': contains information about how to parse the dataset (e.g. window length, which tags to read);
        - 'train_options' contains information about training parameters (e.g. learning rate).

    Examples
    --------
    >>> write_config_json(config_path, learning_rate=0.00001, n_filters=64)
    '''

    # specify how to build the model
    model = {
        "n_dense_units": 0, # number of neurons in the dense hidden layer of the backend
        "n_filters": 0,     # number of filters in the first convolution layer of the log mel-spectrogram frontend (see https://github.com/jordipons/music-audio-tagging-at-scale-models)
    }

    # specify how to train the model
    model_training = {
        "optimizer": {
            "name": "Adam",      # name of the optimizer, as appears in tf.keras.optimizers
            "learning_rate": 0.  # initial learning rate
        },
        "batch_size": 0,                # global batch size
        "interleave_cycle_length": 0,   # number of input elements that are processed concurrently (when using tf.data.Dataset.interleave)
        "interleave_block_length": 0,   # number of consecutive input elements that are consumed at each cycle (when using tf.data.Dataset.interleave) (see https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave)
        "early_stop_patience": 0,       # the number epochs with 'no improvement' to wait before triggering EarlyStopping (please put None if EarlyStopping is not used)
        "reduceLRoP_patience": 0,       # the number epochs with 'no improvement' to wait before triggering ReduceLROnPlateau and reduce lr by a 'reduceLRoP_factor' (please put None if ReduceLROnPlateau is not used)
        "early_stop_min_delta": 0.,     # the minimum increase in PR-AUC between two consecutive epochs to be considered as 'improvment'
        "reduceLRoP_min_delta": 0.,     # the minimum increase in PR-AUC between two consecutive epochs to be considered as 'improvment'
        "reduceLRoP_min_lr": 0.,        # the lower bound for the learning rate, when using ReduceLROnPlateau callback
        "reduceLRoP_factor": 0.,        # the factor the learning rate is deacreased by at each step, when using ReduceLROnPlateau callback
        "log_dir": "~/",                # directory where tensorboard logs and checkpoints will be stored
        "shuffle": True,                # if True, shuffle the dataset
        "shuffle_buffer_size": 0,       # buffer size to use to shuffle the dataset (only applies if shuffle is True)
        "split": MyJSONEnc_NoIndent([0, 0]),      # number of (or percentage of) .tfrecord files that will go in each train/validation/test dataset (ideally an array of len <= 3)
        "window_length": 0,             # length (in seconds) of the audio 'window' to input into the model
        "window_random": True,          # if True, the window is picked randomly along the track length; if False, the window is always picked from the middle
    }

    # specify which tags to use
    tags = {
        "top": 0,                   # e.g. use only the most popular 50 tags from the tags database will go into training (if None, all tags go into training)
        "with": MyJSONEnc_NoIndent([]),       # tags that will be added to the list above        
        "without": MyJSONEnc_NoIndent([]),    # tags that will be excluded from the list above
        "merge": None,              # tags to merge together (e.g. use 'merge': [[1,2], [3,4]] to merge tags 1 and 2, 3 and 4)
    }

    # specify how the data has been encoded in the .tfrecord files
    tfrecords = {
        "n_mels": 0,        # number of mels in the log-mel-spectrogram audio files
        "n_tags": 0,        # *total* number of tags in the database (*not* the number of tags that you will eventually be using for training)
        "sample_rate": 0,   # sample rate of the audio files
    }

    def substitute_into_dict(key, value):
        for dict in (model, model_training, tags, tfrecords):
            if key in dict:
                dict[key] = value
                return
        raise KeyError(key)
    
    # substitute kwargs into output dictionary (passing kwargs is basically equivalent to editing the .json file manually)
    for key, value in kwargs.items():
        substitute_into_dict(key, value)
    
    if os.path.isdir(config_path):
        config_path = os.path.join(os.path.abspath(config_path), 'config.json')
    
    with open(config_path, 'w') as f:
        d = {'model': model, 'model-training': model_training, 'tags': tags, 'tfrecords': tfrecords}
        s = json.dumps(d, cls=MyJSONEnc, indent=2)
        f.write(s)

def parse_config_json(config_path, lastfm):
    ''' Parse a JSON configuration file into a handy Namespace.

    Parameters
    -----------
    config_path: str
        The path to the .json file, or the directory where it is saved.

    lastfm: LastFm, LastFm2Pandas, str
        Instance of the tags database. If a string is passed, try to instantiate the tags database from the (string as a) path.
        
    Returns
    -------
    config: argparse.Namespace
    '''

    if not isinstance(lastfm, object):
        lastfm = lastfm.LastFm(os.path.expanduser(lastfm))

    # if config_path is a folder, assume the folder contains a config.json
    if os.path.isdir(os.path.expanduser(config_path)):
        path = os.path.join(os.path.abspath(os.path.expanduser(config_path)), 'config.json')
    else:
        path = os.path.expanduser(config_path)

    # load json
    with open(path, 'r') as f:
        config_dict = json.loads(f.read())

    # create config namespace
    config = argparse.Namespace(**config_dict['model'], **config_dict['model-training'], **config_dict['tfrecords'])
    config.path = os.path.abspath(config_path)

    # update config (optimizer will be instantiated with tf.get_optimizer using {"class_name": config.optimizer_name, "config": config.optimizer})
    config.optimizer_name = config.optimizer.pop('name')

    # read tags from popularity dataframe
    top = config_dict['tags']['top']
    if (top is not None) and (top !=config.n_tags):
        top_tags = lastfm.popularity()['tag'][:top].tolist()
        tags = set(top_tags)
    else:
        tags=None

    # find tags to use
    if tags is not None:
        if config_dict['tags']['with']:
            tags.update(config_dict['tags']['with'])
        
        if config_dict['tags']['without']:
            tags.difference_update(config_dict['tags']['without'])
        tags = list(tags)
    else:
        raise ValueError("parameter 'with' is inconsistent to parameter 'top'")
    
    config.n_output_neurons = len(tags) if tags is not None else config.n_tags
    config.tags = lastfm.tag_to_tag_num(tags) if tags is not None else None
    config.tags_to_merge = lastfm.tag_to_tag_num(config_dict['tags']['merge']) if config_dict['tags']['merge'] else None

    config.tags = np.sort(config.tags)
    
    return config
    
def frontend_wave(input):
    ''' Create the frontend model for waveform input. '''

    initializer = tf.keras.initializers.VarianceScaling()
    
    input = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 2), name='expdim_1_wave')(input)

    conv0 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=3, padding='valid',
                   activation='relu', kernel_initializer=initializer, name='conv0_wave')(input)
    bn_conv0 = tf.keras.layers.BatchNormalization(name='bn0_wave')(conv0)

    conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer,
                   name='conv1_wave')(bn_conv0)
    bn_conv1 = tf.keras.layers.BatchNormalization(name='bn1_wave')(conv1)
    pool1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=3, name='pool1_wave')(bn_conv1)
    
    conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer,
                   name='conv2_wave')(pool1)
    bn_conv2 = tf.keras.layers.BatchNormalization(name='bn2_wave')(conv2)
    pool2 = tf.keras.layers.MaxPool1D(pool_size=3, strides=3, name='pool2_wave')(bn_conv2)

    conv3 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer,
                   name='conv3_wave')(pool2) 
    bn_conv3 = tf.keras.layers.BatchNormalization(name='bn3_wave')(conv3)
    pool3 = tf.keras.layers.MaxPool1D(pool_size=3, strides=3, name='pool3_wave')(bn_conv3)

    conv4 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer,
                   name='conv4_wave')(pool3) 
    bn_conv4 = tf.keras.layers.BatchNormalization(name='bn4_wave')(conv4)
    pool4 = tf.keras.layers.MaxPool1D(pool_size=3, strides=3, name='pool4_wave')(bn_conv4)
            
    conv5 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer,
                   name='conv5_wave')(pool4)
    bn_conv5 = tf.keras.layers.BatchNormalization(name='bn5_wave')(conv5)
    pool5 = tf.keras.layers.MaxPool1D(pool_size=3, strides=3, name='pool5_wave')(bn_conv5)
            
    conv6 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer=initializer,
                   name='conv6_wave')(pool5)
    bn_conv6 = tf.keras.layers.BatchNormalization(name='bn6_wave')(conv6)
    pool6 = tf.keras.layers.MaxPool1D(pool_size=3, strides=3, name='pool6_wave')(bn_conv6)
    
    exp_dim = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, [3]), name='expdim2_wave')(pool6)
    return exp_dim

def frontend_log_mel_spect(input, y_input=96, num_filts=32):
    ''' Create the frontend model for log-mel-spectrogram input. '''
    
    initializer = tf.keras.initializers.VarianceScaling()
    input = tf.expand_dims(input, 3)
    
    input_pad_7 = tf.keras.layers.ZeroPadding2D(((0, 0), (3, 3)), name='pad7_spec')(input)
    input_pad_3 = tf.keras.layers.ZeroPadding2D(((0, 0), (1, 1)), name='pad3_spec')(input)
    
    # [TIMBRE] filter shape: 0.9y*7
    conv1 = tf.keras.layers.Conv2D(filters=num_filts, 
               kernel_size=[int(0.9 * y_input), 7],
               padding='valid', activation='relu',
               kernel_initializer=initializer, name='conv1_spec')(input_pad_7)    
    bn_conv1 = tf.keras.layers.BatchNormalization(name='bn1_spec')(conv1)
    pool1 = tf.keras.layers.MaxPool2D(pool_size=[int(conv1.shape[1]), 1], 
                      strides=[int(conv1.shape[1]), 1], name='pool1_spec')(bn_conv1)
    p1 = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1), name='sque1_spec')(pool1)
    
    # [TIMBRE] filter shape: 0.9y*3
    conv2 = tf.keras.layers.Conv2D(filters=num_filts*2,
               kernel_size=[int(0.9 * y_input), 3],
               padding='valid', activation='relu',
               kernel_initializer=initializer, name='conv2_spec')(input_pad_3)
    bn_conv2 = tf.keras.layers.BatchNormalization(name='bn2_spec')(conv2)
    pool2 = tf.keras.layers.MaxPool2D(pool_size=[int(conv2.shape[1]), 1], 
                      strides=[int(conv2.shape[1]), 1], name='pool2_spec')(bn_conv2)
    p2 = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1), name='sque2_spec')(pool2)
    
    # [TIMBRE] filter shape: 0.9y*1
    conv3 = tf.keras.layers.Conv2D(filters=num_filts*4,
               kernel_size=[int(0.9 * y_input), 1], 
               padding='valid', activation='relu',
               kernel_initializer=initializer, name='conv3_spec')(input)
    bn_conv3 = tf.keras.layers.BatchNormalization(name='bn3_spec')(conv3)
    pool3 = tf.keras.layers.MaxPool2D(pool_size=[int(conv3.shape[1]), 1], 
                      strides=[int(conv3.shape[1]), 1], name='pool3_spec')(bn_conv3)
    p3 = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1), name='sque3_spec')(pool3)

    # [TIMBRE] filter shape: 0.4y*7
    conv4 = tf.keras.layers.Conv2D(filters=num_filts,
               kernel_size=[int(0.4 * y_input), 7],
               padding='valid', activation='relu',
               kernel_initializer=initializer, name='conv4_spec')(input_pad_7)
    bn_conv4 = tf.keras.layers.BatchNormalization(name='bn4_spec')(conv4)
    pool4 = tf.keras.layers.MaxPool2D(pool_size=[int(conv4.shape[1]), 1], 
                  strides=[int(conv4.shape[1]), 1], name='pool4_spec')(bn_conv4)
    p4 = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1), name='sque4_spec')(pool4)

    # [TIMBRE] filter shape: 0.4y*3
    conv5 = tf.keras.layers.Conv2D(filters=num_filts*2,
               kernel_size=[int(0.4 * y_input), 3],
               padding='valid', activation='relu',
               kernel_initializer=initializer, name='conv5_spec')(input_pad_3)
    bn_conv5 = tf.keras.layers.BatchNormalization(name='bn5_spec')(conv5)
    pool5 = tf.keras.layers.MaxPool2D(pool_size=[int(conv5.shape[1]), 1], 
                      strides=[int(conv5.shape[1]), 1], name='pool5_spec')(bn_conv5)
    p5 = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1), name='sque5_spec')(pool5)

    # [TIMBRE] filter shape: 0.4y*1
    conv6 = tf.keras.layers.Conv2D(filters=num_filts*4,
               kernel_size=[int(0.4 * y_input), 1],
               padding='valid', activation='relu',
               kernel_initializer=initializer, name='conv6_spec')(input)
    bn_conv6 = tf.keras.layers.BatchNormalization(name='bn6_spec')(conv6)
    pool6 = tf.keras.layers.MaxPool2D(pool_size=[int(conv6.shape[1]), 1], 
                  strides=[int(conv6.shape[1]), 1], name='pool6_spec')(bn_conv6)
    p6 = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1), name='sque6_spec')(pool6)

    # Avarage pooling frequency axis
    avg_pool = tf.keras.layers.AveragePooling2D(pool_size=[y_input, 1], 
                             strides=[y_input, 1], name='avgpool_spec')(input)
    avg_pool = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1), name='sque7_spec')(avg_pool)

    # [TEMPORAL] filter shape: 165*1
    conv7 = tf.keras.layers.Conv1D(filters=num_filts, kernel_size=165,
                   padding='same', activation='relu',
                   kernel_initializer=initializer, name='conv7_spec')(avg_pool)
    bn_conv7 = tf.keras.layers.BatchNormalization(name='bn7_spec')(conv7)
    
    # [TEMPORAL] filter shape: 128*1
    conv8 = tf.keras.layers.Conv1D(filters=num_filts*2, kernel_size=128,
                   padding='same', activation='relu',
                   kernel_initializer=initializer, name='conv8_spec')(avg_pool)
    bn_conv8 = tf.keras.layers.BatchNormalization(name='bn8_spec')(conv8)

    # [TEMPORAL] filter shape: 64*1
    conv9 = tf.keras.layers.Conv1D(filters=num_filts*4, kernel_size=64,
                   padding='same', activation='relu',
                   kernel_initializer=initializer, name='conv9_spec')(avg_pool)
    bn_conv9 = tf.keras.layers.BatchNormalization(name='bn9_spec')(conv9)
    
    # [TEMPORAL] filter shape: 32*1
    conv10 = tf.keras.layers.Conv1D(filters=num_filts*8, kernel_size=32,
                   padding='same', activation='relu',
                   kernel_initializer=initializer, name='conv10_spec')(avg_pool)
    bn_conv10 = tf.keras.layers.BatchNormalization(name='bn10_spec')(conv10)
    
    concat = tf.keras.layers.Concatenate(2, name='concat_spec')([p1, p2, p3, p4, p5, p6, bn_conv7, bn_conv8,
                          bn_conv9, bn_conv10])
    
    exp_dim = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 3), name='expdim1_spec')(concat)
    return exp_dim

def backend(input, num_output_neurons, num_units=1024):
    ''' Create the backend model. '''
    
    initializer = tf.keras.initializers.VarianceScaling()
    
    conv1 = tf.keras.layers.Conv2D(filters=512, kernel_size=[7, int(input.shape[2])],
                   padding='valid', activation='relu',
                   kernel_initializer=initializer, name='conv1_back')(input)
    bn_conv1 = tf.keras.layers.BatchNormalization(name='bn1_back')(conv1)
    bn_conv1_t = tf.keras.layers.Permute((1, 3, 2), name='perm1_back')(bn_conv1)
    
    bn_conv1_pad = tf.keras.layers.ZeroPadding2D(((3, 3), (0, 0)), name='pad3_1_back')(bn_conv1_t)
    conv2 = tf.keras.layers.Conv2D(filters=512, kernel_size=[7, int(bn_conv1_pad.shape[2])],
                   padding='valid', activation='relu',
                   kernel_initializer=initializer, name='conv2_back')(bn_conv1_pad)
    conv2_t = tf.keras.layers.Permute((1,3,2), name='perm2_back')(conv2)
    bn_conv2 = tf.keras.layers.BatchNormalization(name='bn2_back')(conv2_t)
    res_conv2 = tf.keras.layers.Add(name='add1_back')([bn_conv2, bn_conv1_t])
    
    pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 1], strides=[2, 1], name='pool1_back')(res_conv2)
    
    pool1_pad = tf.keras.layers.ZeroPadding2D(((3, 3), (0, 0)), name='pad3_2_back')(pool1)
    conv3 = tf.keras.layers.Conv2D(filters=512, kernel_size=[7, int(pool1_pad.shape[2])],
                   padding='valid', activation='relu',
                   kernel_initializer=initializer, name='conv3_back')(pool1_pad)
    conv3_t = tf.keras.layers.Permute((1, 3, 2), name='perm3_back')(conv3)
    bn_conv3 = tf.keras.layers.BatchNormalization(name='bn3_back')(conv3_t)
    res_conv3 = tf.keras.layers.Add(name='add2_back')([bn_conv3, pool1])
    
    max_pool2 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=1), name='glo_max_back')(res_conv3)
    avg_pool2, var_pool2 = tf.keras.layers.Lambda(lambda x: tf.nn.moments(x, axes=[1]), name='moment_back')(res_conv3)
    pool2 = tf.keras.layers.Concatenate(2, name='concat_back')([max_pool2, avg_pool2])
    flat_pool2 = tf.keras.layers.Flatten()(pool2)
    
    flat_pool2_dropout = tf.keras.layers.Dropout(rate=0.5, name='drop1_back')(flat_pool2)
    dense = tf.keras.layers.Dense(units=num_units, activation='relu',
                  kernel_initializer=initializer, name='dense1_back')(flat_pool2_dropout)
    bn_dense = tf.keras.layers.BatchNormalization(name='bn_dense_back')(dense)
    dense_dropout = tf.keras.layers.Dropout(rate=0.5, name='drop2_back')(bn_dense)
    
    return tf.keras.layers.Dense(activation='sigmoid', units=num_output_neurons,
                 kernel_initializer=initializer, name='dense2_back')(dense_dropout)

def build_model(frontend_mode, num_output_neurons=155, y_input=96, num_units=500, num_filts=16, batch_size=None):
    ''' Generate the final model by combining frontend and backend.
    
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
        input = tf.keras.Input(shape=[None], batch_size=batch_size)
        front_out = frontend_wave(input)

    elif frontend_mode == 'log-mel-spectrogram':
        input = tf.keras.Input(shape=[y_input, None], batch_size=batch_size)
        front_out = frontend_log_mel_spect(input, y_input=y_input, num_filts=num_filts)

    else:
        raise ValueError('please specify the frontend_mode: "waveform" or "log-mel-spectrogram"')

    model = tf.keras.Model(input,
                           backend(front_out,
                                   num_output_neurons=num_output_neurons,
                                   num_units=num_units))
    return model

class MyJSONEnc(json.JSONEncoder): # see https://stackoverflow.com/questions/13249415/how-to-implement-custom-indentation-when-pretty-printing-with-the-json-module
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # save copy of any keyword argument values needed for use here
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(MyJSONEnc, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, MyJSONEnc_NoIndent)
                else super(MyJSONEnc, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC
        json_repr = super(MyJSONEnc, self).encode(obj) # default JSON repr

        # replace any marked-up object ids in the JSON repr with the value returned from the json.dumps() of the corresponding wrapped object
        for match in self.regex.finditer(json_repr):
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

            # replace the matched id string with json formatted representation of the corresponding object
            json_repr = json_repr.replace(
                            '"{}"'.format(format_spec.format(id)), json_obj_repr)
        return json_repr

class MyJSONEnc_NoIndent(): # value wrapper
    def __init__(self, value):
        self.value = value