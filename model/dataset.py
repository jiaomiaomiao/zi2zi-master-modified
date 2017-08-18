# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import cPickle as pickle
import numpy as np
import random
import os
from .utils import pad_seq, bytes_to_file, \
    read_split_image, shift_and_resize_image, normalize_image


class PickledImageProvider(object):
    def __init__(self, obj_path,train_mark=False):
        self.obj_path = obj_path
        self.examples = self.load_pickled_examples(train_mark)

    def load_pickled_examples(self,train_mark=False):
        with open(self.obj_path, "rb") as of:
            examples = list()
            while True:
                try:
                    e = pickle.load(of)
                    examples.append(e)
                    if len(examples) % 10000 == 0:
                        if train_mark==True:
                            print("processed %d examples for train" % len(examples))
                        else:
                            print("processed %d examples for val" % len(examples))
                except EOFError:
                    break
                except Exception:
                    pass
            if train_mark==True:
                print("unpickled total %d examples for train" % len(examples))
            else:
                print("unpickled total %d examples for val" % len(examples))
            return examples


def get_batch_iter(examples, batch_size, augment):
    # the transpose ops requires deterministic
    # batch size, thus comes the padding
    padded = pad_seq(examples, batch_size)

    def process(img):
        img = bytes_to_file(img)
        try:
            img_A, img_B = read_split_image(img)
            if augment:
                # augment the image by:
                # 1) enlarge the image
                # 2) random crop the image back to its original size
                # NOTE: image A and B needs to be in sync as how much
                # to be shifted
                w, h, _ = img_A.shape
                multiplier = random.uniform(1.00, 1.20)
                # add an eps to prevent cropping issue
                nw = int(multiplier * w) + 1
                nh = int(multiplier * h) + 1
                shift_x = int(np.ceil(np.random.uniform(0.01, nw - w)))
                shift_y = int(np.ceil(np.random.uniform(0.01, nh - h)))
                img_A = shift_and_resize_image(img_A, shift_x, shift_y, nw, nh)
                img_B = shift_and_resize_image(img_B, shift_x, shift_y, nw, nh)
            img_A = normalize_image(img_A) #target
            img_B = normalize_image(img_B) #source
            return np.concatenate([img_A, img_B], axis=2)
        finally:
            img.close()

    def batch_iter():
        for i in range(0, len(padded), batch_size):
            batch = padded[i: i + batch_size]
            labels = [e[0] for e in batch]
            processed = [process(e[1]) for e in batch]
            # stack into tensor
            yield labels, np.array(processed).astype(np.float32)

    return batch_iter()


class TrainDataProvider(object):
    def __init__(self,
                 data_dir='./',
                 train_name="train.obj", val_name="val.obj", infer_name="infer.obj",
                 filter_by=None, sub_train_set_num=-1,
                 infer_mark=False):
        self.data_dir = data_dir
        self.filter_by = filter_by
        if not infer_mark:

            self.train = PickledImageProvider(train_name, train_mark=True)
            self.val = PickledImageProvider(val_name, train_mark=False)

            if self.filter_by:
                print("filter by label ->", filter_by)
                self.train.examples = filter(lambda e: e[0] in self.filter_by, self.train.examples)
                self.val.examples = filter(lambda e: e[0] in self.filter_by, self.val.examples)

            if not sub_train_set_num == -1:
                def diff(first, second):
                    second = set(second)
                    return [item for item in first if item not in second]

                print("sub training set: @ %d for each label are trained" % (sub_train_set_num))
                full_indices = range(len(self.train.examples))
                selected_indices = random.sample(full_indices, sub_train_set_num * len(self.filter_by))
                non_selected_indices = diff(full_indices, selected_indices)

                new_train_examples = list()
                for indices in selected_indices:
                    new_train_examples.append(self.train.examples[indices])
                new_val_examples = list()
                for indices in non_selected_indices:
                    new_val_examples.append(self.train.examples[indices])
                self.train.examples = list()
                self.train.examples = new_train_examples
                self.val.examples.extend(new_val_examples)

            print("in total train examples -> %d, val examples -> %d" % (
            len(self.train.examples), len(self.val.examples)))
        else:
            self.infer = PickledImageProvider(infer_name, train_mark=False)
            if self.filter_by:
                print("filter by label ->", filter_by)
                self.infer.examples = filter(lambda e: e[0] in self.filter_by, self.infer.examples)
            print("in total infer examples -> %d" % (len(self.infer.examples)))






    def get_total_epoch_num(self,itr_num=-1,batch_size=-1,tower_num=-1):
        epoch_num = int(np.ceil(itr_num / np.ceil(len(self.train.examples) / (batch_size*tower_num))))
        print ("Epoch Num:%d, Itr Num:%d" % (epoch_num, itr_num) )
        return epoch_num


    def get_train_iter(self, batch_size, shuffle=True):
        training_examples = self.train.examples[:]
        if shuffle:
            np.random.shuffle(training_examples)
        return get_batch_iter(training_examples, batch_size, augment=True)

    def get_val_iter(self, batch_size, shuffle=True):
        """
        Validation iterator runs forever
        """
        val_examples = self.val.examples[:]
        if shuffle:
            np.random.shuffle(val_examples)
        while True:
            val_batch_iter = get_batch_iter(val_examples, batch_size, augment=False)
            for labels, examples in val_batch_iter:
                yield labels, examples

    def get_infer_iter(self, batch_size, shuffle=False):
        """
        Validation iterator runs forever
        """
        infer_examples = self.infer.examples[:]
        if shuffle:
            np.random.shuffle(infer_examples)
        while True:
            infer_batch_iter = get_batch_iter(infer_examples, batch_size, augment=False)
            for labels, examples in infer_batch_iter:
                yield labels, examples

    def compute_total_batch_num(self, batch_size,tower_num):
        """Total padded batch num"""
        return int(np.ceil(len(self.train.examples) / float(batch_size*tower_num)))


class InjectDataProvider(object):
    def __init__(self, obj_path):
        self.data = PickledImageProvider(obj_path)
        print("examples -> %d" % len(self.data.examples))

    def get_single_embedding_iter(self, batch_size, embedding_id):
        examples = self.data.examples[:]
        batch_iter = get_batch_iter(examples, batch_size, augment=False)
        for _, images in batch_iter:
            # inject specific embedding style here
            labels = [embedding_id] * batch_size
            yield labels, images

    def get_random_embedding_iter(self, batch_size, embedding_ids):
        examples = self.data.examples[:]
        batch_iter = get_batch_iter(examples, batch_size, augment=False)
        for _, images in batch_iter:
            # inject specific embedding style here
            # by randomly generating labels
            labels = [random.choice(embedding_ids) for i in range(batch_size)]
            yield labels, images
