# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Text iterator
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import random
import math


class TextIterator:
    """Create text iterator for sequence pair classification problem. 
    Data file is assumed to contain one sample per line. The format is 
    label\tsequence1\tsequence2. 
    Args:
        input_file: path of the input text file.
        token_to_idx: a dictionary, which convert token to index
        batch_size: mini-batch size 
        vocab_size: limit on the size of the vocabulary, if token index is 
            larger than vocab_size, return UNK (index 1)
        shuffle: Boolean; if true, we will first sort a buffer of samples by 
            sequence length, and then shuffle it by batch-level.
        factor: buffer size is factor * batch-size 

    """

    def __init__(self, input_file, token_to_idx,
                 batch_size=128, vocab_size=-1, shuffle=True, factor=20):
        self.input_file = open(input_file, 'r')
        self.token_to_idx = token_to_idx
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.shuffle = shuffle
        self.end_of_data = False
        self.instance_buffer = []
        # buffer for shuffle
        self.max_buffer_size = batch_size * factor

    def __iter__(self):
        return self

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.input_file.seek(0)
            raise StopIteration

        instance = []

        if len(self.instance_buffer) == 0:
            for _ in range(self.max_buffer_size):
                line = self.input_file.readline()
                if line == "":
                    break
                arr = line.strip().split('\t')
                assert len(arr) == 3
                self.instance_buffer.append(
                    [arr[0], arr[1].split(' '), arr[2].split(' ')])

            if self.shuffle:
                # sort by length of sum of target buffer and target_buffer
                length_list = []
                for ins in self.instance_buffer:
                    current_length = len(ins[1]) + len(ins[2])
                    length_list.append(current_length)

                length_array = numpy.array(length_list)
                length_idx = length_array.argsort()
                # shuffle mini-batch
                tindex = []
                small_index = range(
                    int(math.ceil(len(length_idx) * 1. / self.batch_size)))
                random.shuffle(small_index)
                for i in small_index:
                    if (i + 1) * self.batch_size > len(length_idx):
                        tindex.extend(length_idx[i * self.batch_size:])
                    else:
                        tindex.extend(
                            length_idx[i * self.batch_size:(i + 1) * self.batch_size])

                _buf = [self.instance_buffer[i] for i in tindex]
                self.instance_buffer = _buf

        if len(self.instance_buffer) == 0:
            self.end_of_data = False
            self.input_file.seek(0)
            raise StopIteration

        try:
            # actual work here
            while True:
                # read from source file and map to word index
                try:
                    current_instance = self.instance_buffer.pop(0)
                except IndexError:
                    break

                label = current_instance[0]
                sent1 = current_instance[1]
                sent2 = current_instance[2]

                sent1.insert(0, '_BOS_')
                sent1.append('_EOS_')
                sent1 = [self.token_to_idx[w]
                         if w in self.token_to_idx else 1 for w in sent1]
                if self.vocab_size > 0:
                    sent1 = [w if w < self.vocab_size else 1 for w in sent1]

                sent2.insert(0, '_BOS_')
                sent2.append('_EOS_')
                sent2 = [self.token_to_idx[w] if w in self.token_to_idx else 1
                         for w in sent2]
                if self.vocab_size > 0:
                    sent2 = [w if w < self.vocab_size else 1 for w in sent2]

                instance.append([label, sent1, sent2])

                if len(instance) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(instance) <= 0:
            self.end_of_data = False
            self.input_file.seek(0)
            raise StopIteration

        return instance
