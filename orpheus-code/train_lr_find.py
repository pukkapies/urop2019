import numpy as np
import tensorflow as tf

from data_input import generate_datasets_from_dir
from orpheus_model import build_model
from orpheus_model import parse_config_json

def lr_find(dataset, strategy, config, frontend='log-mel-spectrogram', start_lr=1e-07, end_lr=10, num_it=10000, stop_div=True):
    with strategy.scope():
        model = build_model(frontend, num_output_neurons=config.num_output_neurons, num_dense_units=config.num_dense_units, y_input=config.melspect_y)
        
        optimizer = tf.keras.optimizers.get({"class_name": config.optimizer_name, "config": config.optimizer})

        train_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        # train_mean_loss = tf.keras.metrics.Mean(name='train_mean_loss', dtype=tf.float32)
    
        @tf.function
        def train_step(batch):
            # unpack audio and track labels (i.e. tags)
            batch_x, batch_y = batch

            # calculate gradients used to optimize the model's variables
            with tf.GradientTape() as tape:
                loss = train_loss(batch_y,  model(batch_x)) / config.batch_size
            grads = tape.gradient(loss, model.trainable_variables)
            
            # update the model's variables
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            return loss

        # initialize iteration count and lr
        it = 0
        lr = start_lr
        
        f = np.power(end_lr/start_lr, 1/num_it) # the factor to multiply lr by after each iteration

        for batch in dataset:
            it += 1
            lr *= f
            optimizer.learning_rate.assign(lr)

            per_replica_loss = strategy.experimental_run_v2(train_step, args=(batch, ))

            print(it, lr, per_replica_loss)

            if it >= num_it:
                break