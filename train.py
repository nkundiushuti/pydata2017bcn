import os
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras import backend as K
from keras.backend import tensorflow_backend as KTF
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from model import build_model, get_session

from settings import *

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

init_lr = 0.001
dataset_mean = np.load("./IRMAS-Sample/data_mean.npy")
global_namescope = 'train'
in_memory_data = True
KTF.set_session(get_session(gpu_fraction=0.4))


class TensorEmbeddings(Callback):
    """Tensorboard Embeddings visualization callback."""

    def __init__(self, log_dir='./logs_embed',
                 batch_size=32,
                 freq=0,
                 layer_names=None,
                 metadata=None,
                 sprite=None,
                 sprite_shape=None):
        super(TensorEmbeddings, self).__init__()
        self.log_dir = log_dir
        self.freq = freq
        self.layer_names = layer_names
        self.metadata = metadata
        self.sprite = sprite
        self.sprite_shape = sprite_shape
        self.batch_size = batch_size

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        self.summary_writer = tf.summary.FileWriter(self.log_dir)
        self.embeddings_ckpt_path = os.path.join(self.log_dir, 'keras_embedding.ckpt')

        if self.freq:
            embeddings_layers = [layer for layer in self.model.layers
                                 if layer.name in self.layer_names]
            self.output_tensors = [tf.get_default_graph().get_tensor_by_name(layer.get_output_at(0).name)
                                   for layer in embeddings_layers]

            config = projector.ProjectorConfig()
            for i in range(len(self.output_tensors)):
                embedding = config.embeddings.add()
                embedding.tensor_name = '{ns}/embedding_{i}'.format(ns=global_namescope, i=i)

                # Simpliest metadata handler, a single file for all embeddings
                if self.metadata:
                    embedding.metadata_path = self.metadata

                # Sprite image handler
                if self.sprite and self.sprite_shape:
                    embedding.sprite.image_path = self.sprite
                    embedding.sprite.single_image_dim.extend(self.sprite_shape)

            self.embedding_vars = [tf.Variable(np.zeros((len(self.validation_data[0]),
                                                         self.output_tensors[i].shape[1]),
                                                        dtype='float32'),
                                               name='embedding_{}'.format(i))
                                   for i in range(len(self.output_tensors))]
            for embedding_var in self.embedding_vars:
                self.sess.run(embedding_var.initializer)

            projector.visualize_embeddings(self.summary_writer, config)
            self.saver = tf.train.Saver(self.embedding_vars)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data and self.freq:
            if epoch % self.freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)
                all_embeddings = [[]]*len(self.output_tensors)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    batch_val = []
                    batch_val.append(val_data[0][i:i + step])
                    batch_val.append(val_data[1][i:i + step])
                    batch_val.append(val_data[2][i:i + step])
                    if self.model.uses_learning_phase:
                        batch_val.append(val_data[3])
                    feed_dict = dict(zip(tensors, batch_val))
                    tensor_outputs = self.sess.run(self.output_tensors, feed_dict=feed_dict)
                    for output_idx, tensor_output in enumerate(tensor_outputs):
                        all_embeddings[output_idx].extend(tensor_output)
                    i += self.batch_size
                for embedding_idx, embed in enumerate(self.embedding_vars):
                    embed.assign(np.array(all_embeddings[embedding_idx])).eval(session=self.sess)
                self.saver.save(self.sess, self.embeddings_ckpt_path, epoch)

        self.summary_writer.flush()

    def on_train_end(self, _):
        self.summary_writer.close()


def _get_extended_data(inputs, targets):
    extended_inputs = list()
    for i in range(0, N_SEGMENTS_PER_TRAINING_FILE):
        extended_inputs.extend(['_'.join(list(x)) for x in zip(inputs, [str(i)]*len(inputs))])
    extended_inputs = np.array(extended_inputs)
    extended_targets = np.tile(np.array(targets).reshape(-1),
                               N_SEGMENTS_PER_TRAINING_FILE).reshape(-1, IRMAS_N_CLASSES)
    return extended_inputs, extended_targets


def _load_features(filenames):
    features = list()
    for filename in filenames:
        feature_filename = os.path.join(IRMAS_TRAIN_FEATURE_BASEPATH,
                                        "{}.npy".format(filename))
        feature = np.load(feature_filename)
        feature -= dataset_mean
        features.append(feature)

    features = np.array(features).reshape(-1, N_MEL_BANDS, SEGMENT_DUR, 1)
    return features


def _batch_generator(inputs, targets):
    assert len(inputs) == len(targets)
    while True:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - BATCH_SIZE + 1, BATCH_SIZE):
            excerpt = indices[start_idx:start_idx + BATCH_SIZE]
            if in_memory_data:
                yield inputs[excerpt], targets[excerpt]
            else:
                yield _load_features(inputs[excerpt]), targets[excerpt]


def _solid_full_data(inputs, targets):
    if in_memory_data:
        return inputs, targets
    return _load_features(inputs), targets


def train(X_train, X_val, y_train, y_val):
    with tf.name_scope('optimizer'):
        optimizer = SGD(lr=init_lr, momentum=0.9, nesterov=True)

    with tf.name_scope('model'):
        model = build_model(IRMAS_N_CLASSES)

    with tf.name_scope('callbacks'):
        early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_EPOCH)
        save_clb = ModelCheckpoint(
            "{weights_basepath}/".format(
                weights_basepath=MODEL_WEIGHT_BASEPATH) +
            "epoch.{epoch:02d}-val_loss.{val_loss:.3f}",
            monitor='val_loss',
            save_best_only=True)
        embeddings_to_monitor = ['hidden']
        metadata_file_name = 'metadata.tsv'
        sprite_file_name = 'sprite.png'
        embeddings_metadata = metadata_file_name
        tb = TensorBoard(log_dir='./logs', histogram_freq=1,
                         write_graph=True, write_grads=True, write_images=True)
        tb.validation_data = _solid_full_data(X_val, y_val)
        tbe = TensorEmbeddings(log_dir='./logs_embed', freq=1,
                               layer_names=embeddings_to_monitor,
                               metadata=embeddings_metadata,
                               sprite=sprite_file_name,
                               sprite_shape=[N_MEL_BANDS, SEGMENT_DUR])
        tbe.validation_data = _solid_full_data(X_val, y_val)
    model.summary()

    with tf.name_scope('compile'):
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    with tf.name_scope(global_namescope):
        model.fit_generator(_batch_generator(X_train, y_train),
                            steps_per_epoch=STEPS_PER_EPOCH,
                            nb_epoch=MAX_EPOCH_NUM,
                            verbose=2,
                            callbacks=[save_clb, early_stopping, tb, tbe],
                            validation_data=_solid_full_data(X_val, y_val),
                            class_weight=None,
                            nb_worker=1)


def main():
    dataset = pd.read_csv(IRMAS_TRAINING_META_PATH, names=["filename", "class_id"])
    X_train, X_val, y_train, y_val = train_test_split(list(dataset.filename),
                                                      to_categorical(np.array(dataset.class_id, dtype=int)),
                                                      test_size=VALIDATION_SPLIT, random_state=3)
    extended_x_train, extended_y_train = _get_extended_data(X_train, y_train)
    extended_x_val, extended_y_val = _get_extended_data(X_val, y_val)
    y_train = extended_y_train
    y_val = extended_y_val
    X_train = _load_features(extended_x_train)
    X_val = _load_features(extended_x_val)

    train(X_train, X_val, y_train, y_val)

if __name__ == "__main__":
    main()
