import os
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from model import build_model

from settings import *

init_lr = 0.001
dataset_mean = np.load("./IRMAS-Sample/data_mean.npy")
in_memory_data = True

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
    optimizer = SGD(lr=init_lr, momentum=0.9, nesterov=True)

    model = build_model(IRMAS_N_CLASSES)

    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_EPOCH)
    save_clb = ModelCheckpoint(
        "{weights_basepath}/".format(
            weights_basepath=MODEL_WEIGHT_BASEPATH) +
        "epoch.{epoch:02d}-val_loss.{val_loss:.3f}",
        monitor='val_loss',
        save_best_only=True)
    tb = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_grads=True)
    tb.validation_data = _solid_full_data(X_val, y_val)

    model.summary()
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(_batch_generator(X_train, y_train),
                        samples_per_epoch=SAMPLES_PER_EPOCH,
                        nb_epoch=MAX_EPOCH_NUM,
                        verbose=2,
                        callbacks=[save_clb, early_stopping, tb],
                        validation_data=_solid_full_data(X_val, y_val),
                        nb_val_samples=SAMPLES_PER_VALIDATION,
                        class_weight=None,
                        nb_worker=1)


def main():
    dataset = pd.read_csv(IRMAS_TRAINING_META_PATH, names=["filename", "class_id"])
    X_train, X_val, y_train, y_val = train_test_split(list(dataset.filename),
                                                      to_categorical(np.array(dataset.class_id, dtype=int)),
                                                      test_size=VALIDATION_SPLIT)
    extended_x_train, extended_y_train = _get_extended_data(X_train, y_train)
    extended_x_val, extended_y_val = _get_extended_data(X_val, y_val)
    y_train = extended_y_train
    y_val = extended_y_val
    X_train = _load_features(extended_x_train)
    X_val = _load_features(extended_x_val)

    train(X_train, X_val, y_train, y_val)

if __name__ == "__main__":
    main()
