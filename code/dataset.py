# -*- coding:utf-8 -*-
import pickle
import numpy as np
import collections

DataSets = collections.namedtuple('DataSets', ['train', 'validation', 'test'])


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
    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)
    return DataSets(train=train, validation=validation, test=test)


class DataSet(object):

    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
        labels = np.array(list(map(deal_labels, [x for x in labels])))
        self._labels = labels.reshape(labels.shape[0], labels.shape[1] * 1)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


if __name__ == '__main__':
    read_data_sets('d:\\python code\\snh\\data\\ocr_test.pkl')


