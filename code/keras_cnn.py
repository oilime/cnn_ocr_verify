# -*- coding:utf-8 -*-
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils.vis_utils import plot_model
import os

nb_filters = 32
nb_filters_1 = 64
pool_size = (2, 2)
kernel_size = (3, 3)
nb_epoch = 20
batch_size = 50


def deal_labels(label):
    data = []
    for i in label:
        data = data + i * [0] + [i] + (9 - i) * [0]
    return data


def read_data_sets(data_path):
    with open(data_path, 'rb') as f:
        tr_d, va_d, te_d = pickle.load(f)
    train_images = np.array([np.reshape(x, (1800, 1)) for x in tr_d[0]])
    train_labels = np.array(tr_d[1])
    validation_images = np.array([np.reshape(x, (1800, 1)) for x in va_d[0]])
    validation_labels = np.array(va_d[1])
    test_images = np.array([np.reshape(x, (1800, 1)) for x in te_d[0]])
    test_labels = np.array(te_d[1])
    return train_images, train_labels, validation_images, validation_labels, test_images, test_labels


def main():
    train_images, train_labels, validation_images, validation_labels, test_images, test_labels = \
        read_data_sets('d:\\python code\\snh\\data\\ocr_test.pkl')
    x_train = train_images.reshape(train_images.shape[0], 30, 60, 1)
    x_val = validation_images.reshape(validation_images.shape[0], 30, 60, 1)
    x_test = test_images.reshape(test_images.shape[0], 30, 60, 1)
    y_train = np.array(list(map(deal_labels, [x for x in train_labels])))
    y_val = np.array(list(map(deal_labels, [x for x in validation_labels])))
    y_test = np.array(list(map(deal_labels, [x for x in test_labels])))

    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size=kernel_size, input_shape=(30, 60, 1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters_1, kernel_size=kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(40))
    model.add(Activation('softmax'))

    model.summary()
    plot_model(model, to_file=os.path.join(os.path.pardir, 'data', 'model.png'), show_shapes=True)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(x_val, y_val))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    main()


