import os
import datetime

import tensorflow as tf

import projectname
import projectname_input

def get_optimizer(config):
    return tf.keras.optimizers.get({"class_name": config.optimizer.pop('name'), "config": config.optimizer})

def get_compiled_model(config, audio_format, checkpoint=None):
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        optimizer = get_optimizer(config)
        model = projectname.build_model(audio_format, config.n_output_neurons, config.n_dense_units, config.n_filters)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM), metrics=[[tf.keras.metrics.AUC(curve='ROC', name='AUC-ROC'), tf.keras.metrics.AUC(curve='PR', name='AUC-PR')]])
        if args.checkpoint:
            model.load_weights(os.path.expanduser(checkpoint))
    return model

def train(model, train_dataset, valid_dataset, epochs, steps_per_epoch=None, update_freq=1):

    log_dir = os.path.expanduser("~/logs/fit/" + datetime.datetime.now().strftime("%y%m%d-%H%M"))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath = os.path.join(log_dir, 'mymodel.h5'),
            monitor = 'val_AUC-ROC',
            mode = 'max',
            save_best_only = True,
            save_freq = 'epoch',
            verbose = 1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor = 'val_AUC-ROC',
            mode = 'max',
            min_delta = 0.2,
            restore_best_weights = True,
            patience = 5,
            verbose = 1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor = 'val_loss',
            mode = 'min',
            min_delta = 0.5,
            min_lr = 0.00001,
            factor = 0.2,
            patience = 2,
            verbose = 1,
        ),

        tf.keras.callbacks.TensorBoard(
            log_dir = log_dir, 
            histogram_freq = 1,
            write_graph = False,
            update_freq = update_freq,
            profile_batch = 0,
        ),

        tf.keras.callbacks.TerminateOnNaN(),
    ]

    history = model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, validation_data=valid_dataset)

    return history

def main(tfrecords_dir, audio_format, config, epochs, steps_per_epoch=None, split=None, checkpoint=None, update_freq=1):
    train_dataset, valid_dataset = projectname_input.generate_datasets_from_dir(tfrecords_dir, audio_format, split, **config.config)

    model = get_compiled_model(config, audio_format, checkpoint)

    history = train(model, train_dataset, valid_dataset, epochs, steps_per_epoch, update_freq)

    return history