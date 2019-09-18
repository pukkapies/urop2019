'''
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
'''

import time
import argparse
import tensorflow as tf
import os
import librosa
import numpy as np
import audioread

def log_mel_spec_frontend(input, y_input=96, num_filts=32):
    ''' Creates the frontend model for log-mel-spectrogram input. '''
    
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
    ''' Creates the backend model. '''
    
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

def build_model(num_output_neurons=155, y_input=96, num_units=500, num_filts=16, batch_size=None):

    input = tf.keras.Input(shape=[y_input, None], batch_size=batch_size)
    front_out = log_mel_spec_frontend(input, y_input=y_input, num_filts=num_filts)

    model = tf.keras.Model(input,
                           backend(front_out,
                                   num_output_neurons=num_output_neurons,
                                   num_units=num_units))
    return model




def generate_config():
    
    all_tags = ['rock', 'female', 'pop', 'alternative', 'male', 'indie', 
            'electronic', '00s', 'rnb', 'dance', 'hip-hop', 'instrumental', 
            'chillout', 'alternative rock', 'jazz', 'metal', 'classic rock', 
            'indie rock', 'rap', 'soul', 'mellow', '90s', 'electronica', '80s', 
            'folk', 'chill', 'funk', 'blues', 'punk', 'hard rock', 'pop rock', 
            '70s', 'ambient', 'experimental', '60s', 'easy listening', 
            'rock n roll', 'country', 'electro', 'punk rock', 'indie pop', 
            'heavy metal', 'classic', 'progressive rock', 'house', 'ballad', 
            'psychedelic', 'synthpop', 'trance', 'trip-hop', 'lounge', 
            'techno', 'post-punk', 'reggae', 'new wave', 'britpop', 
            'blues rock', 'folk rock', 'death metal', 'emo', 'soft rock', 
            'latin', 'electropop', 'progressive', '50s', 'disco', 'industrial', 
            'progressive metal', 'post-rock', 'smooth jazz', 'pop punk', 
            'metalcore', 'thrash metal', 'gothic', 'psychedelic rock', 
            'alt-country', 'club', 'alternative  punk', 'avant-garde', 'ska', 
            'americana', 'nu jazz', 'fusion', 'post-hardcore', 'new age', 
            'power pop', 'nu metal', 'black metal', 'power metal', 'grunge', 
            'acid jazz', 'dub', 'garage rock', 'neo-soul', 
            'melodic death metal', 'underground hip-hop', 'alternative metal', 
            'idm', 'darkwave', 'alt rock', 'gothic metal', 'ethereal', 'swing', 
            'glam rock', 'progressive trance', 'lo-fi', 'rockabilly', 'classical', 
            'metro downtempo', 'dream pop', 'melodic metal', 'doom metal', 'bass', 
            'shoegaze', 'gothic rock', 'heavy', 'dancehall', 'art rock', 
            'classic country', 'screamo', 'christmas', 'hardcore punk', 
            'celtic', 'garage', 'rockpop', 'synth', 'indietronica', 
            'vocal jazz', 'jazz fusion', 'stoner rock', 'jazz vocal', 
            'electro house', 'grindcore', 'vocal trance', 'christian rock', 
            'indie folk', 'ebm', 'old school soul', 'goth', 'southern rock', 
            'progressive house', 'symphonic metal', 'eurodance', 'deep house', 
            'roots reggae', 'gospel', 'industrial metal', 'brutal death metal', 
            'bluegrass', 'minimal techno', 'electroclash', 'salsa', 
            'speed metal', 'thrash', 'experimental rock']
    
    tag_to_tag_num = {}
    tag_num_to_tag = {}
    
    for idx, t in enumerate(all_tags):
        tag_to_tag_num[t] = idx + 1
        tag_num_to_tag[idx+1] = t
    
    #assume top is not None and top != 155
    config_dict = {"model": {"n_dense_units": 500, "n_filters": 16},
              "model-training": {"window_length": 15},
              "tags": {"top": 50, "with": [], "without": [], "merge": None},
              "tfrecords": {"n_mels": 96, "n_tags": len(all_tags), "sample_rate": 16000}}
    
    top = config_dict['tags']['top']
    top_tags = all_tags[:top]
    tags = set(top_tags)
    
    if config_dict['tags']['with']:
        tags.update(config_dict['tags']['with'])
    if config_dict['tags']['without']:
        tags.difference_update(config_dict['tags']['without'])
    
    config = argparse.Namespace()
    config.n_dense_units = config_dict['model']['n_dense_units']
    config.n_filters = config_dict['model']['n_filters']
    config.n_mels = config_dict['tfrecords']['n_mels']
    config.n_output_neurons = len(tags)
    config.sr = config_dict['tfrecords']['sample_rate']
    config.tags = [tag_to_tag_num[t] for t in list(tags)]
    config.tags_to_merge = [[tag_to_tag_num[tag] for tag in tags] for tags in config_dict['tags']['merge']] if config_dict['tags']['merge'] is not None else None
    config.tot_tags = config_dict['tfrecords']['n_tags']
    config.window_len = config_dict['model-training']['window_length']
    config.tag_to_tag_num = tag_to_tag_num
    config.tag_num_to_tag = tag_num_to_tag
    return config
    

def get_audio(config, mp3_path=None, array=None, array_sr=None):
    if mp3_path:
        array, sr_in = librosa.core.load(mp3_path, sr=None, mono=False)
    elif array is not None:
        array = array.astype(np.float32)
        sr_in = array_sr
    array = librosa.core.to_mono(array)
    array = librosa.resample(array, sr_in, config.sr)

    array = librosa.core.power_to_db(librosa.feature.melspectrogram(array, config.sr, n_mels=config.n_mels))
    array = array.astype(np.float32)
    # normalization
    mean, variance = tf.nn.moments(tf.constant(array), axes=[0,1], keepdims=True)
    array = tf.nn.batch_normalization(array, mean, variance, offset = 0, scale = 1, variance_epsilon = .000001).numpy()

    return array


def get_slices(audio, sample_rate, window_size=15):

    slice_length = window_size*sample_rate//512
    n_slices = audio.shape[1]//slice_length

    slices = [audio[:,i*slice_length:(i+1)*slice_length] for i in range(n_slices)]
    slices.append(audio[:,-slice_length:])
    return np.array(slices)



def predict(model, config, audio, cutoff=0.5, window_size=15):

    # make sure tags are sorted
    with_tags = np.sort(config.tags)

    # compute average by using a moving window
    slices = get_slices(audio, config.sr, window_size)
    logits = tf.reduce_mean(model(slices, training=False), axis=[0])
    
    # get tags
    tags = []
    for idx, val in enumerate(logits):
        if val >= cutoff:
            tags.append([float(val.numpy()), config.tag_num_to_tag[int(with_tags[idx])]])
    tags = sorted(tags, key=lambda x:x[0], reverse=True)
    return tags

if __name__ == '__main__':

    parser = argparse.ArgumentParser()     
    parser.add_argument("--checkpoint", help="Path to a checkpoints", default='predict_checkpoint/epoch-18.index')
    parser.add_argument("--mp3-path", help="Path to mp3 dir or mp3 file to predict")
    parser.add_argument("--from-recording", help="If True, the input audio will be recorded from your microphone", action="store_true")
    parser.add_argument("-s", "--recording-second", help="Number of seconds to record. Minimum length is 15 seconds", type=int, default='15')
    parser.add_argument("--cutoff", type=float, help="Lower bound for what prediction values to print", default=0.1)

    args = parser.parse_args()
    print(args)
    
    #load config
    config = generate_config()
    
    # load from checkpoint
    model = build_model(num_output_neurons=config.n_output_neurons, y_input=config.n_mels,
                        num_units=config.n_dense_units, num_filts=config.n_filters)
    
    # restoring from checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    print('Loading from {}'.format(args.checkpoint))
    checkpoint.restore(args.checkpoint)

    if not (args.mp3_path or args.from_recording):
        raise ValueError("If predicting, must either specify mp3 file(s) to predict, or set --from-recording as True")
    elif (args.mp3_path and args.from_recording):
        raise ValueError("If predicting, must either specify mp3 file(s) to predict, or set --from-recording as True")
    elif args.mp3_path:
        if os.path.isfile(args.mp3_path):
            try:
                audio = get_audio(config=config, mp3_path=args.mp3_path)
                print("prediction: ", predict(model, config, audio, cutoff=args.cutoff))
            except audioread.NoBackendError:
                print('skipping {} due to NoBackendError.'.format(args.mp3_path))
            
        else:
            for path in os.listdir(args.mp3_path): 
                try:
                    audio = get_audio(config=config, mp3_path=os.path.join(args.mp3_path, path))
                except audioread.NoBackendError:
                    print('skipping {} due to NoBackendError.'.format(path))
                    continue
                
                print("file: ", path)
                print("prediction: ", predict(model, config, audio, cutoff=args.cutoff))
                print()
                
    else:
        assert args.recording_second >=15
        
        # In case this is not installed automatically
        import sounddevice as sd
        sr_rec = 44100  # Sample rate
        seconds = int(args.recording_second)  # Duration of recording
        while True:
            val = input('Press Enter to begin')
            if val is not None:
                break

        print('record starts in')
        print('3')
        time.sleep(1)
        print('2')
        time.sleep(1)
        print('1')
        time.sleep(1)
        print('0')
        print('Recording')
        audio = sd.rec(int(seconds * sr_rec), samplerate=sr_rec, channels=2)
        sd.wait()  # Wait until recording is finished
        from scipy.io.wavfile import write
        write('hi.wav', sr_rec, audio)  # Save as WAV file

        audio = audio.transpose()
        audio = get_audio(config=config, array=audio, array_sr=sr_rec)
        print("prediction: ", predict(model, config, audio, cutoff=args.cutoff))