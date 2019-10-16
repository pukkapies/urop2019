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

        @tf.function
        def train_step(batch):
            def _train_step(batch):
                features, labels = batch
                with tf.GradientTape() as tape:
                    loss = train_loss(labels,  model(features)) / config.batch_size
                
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                return loss
      
            per_example_losses = strategy.experimental_run_v2(_train_step, args=(batch, ))
            
            mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)
            return mean_loss

        # initialize iteration count and lr
        it = 0
        lr = start_lr
        
        f = np.power(end_lr/start_lr, 1/num_it) # the factor to multiply lr by after each iteration
        
        output = []

        for batch in dataset:
            it += 1
            lr *= f
            optimizer.learning_rate.assign(lr)

            loss = train_step(batch)

            print(it, lr, loss)
            
            output.append((it, lr, loss))

            if it >= num_it:
                break

        return output