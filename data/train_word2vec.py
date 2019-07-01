#!/usr/bin/env python

# Copyright (C) 2019 Alibaba Group Holding Limited
# Copyright (C) 2017 Pan Yang (panyangnlp@gmail.com)

from __future__ import print_function

import logging
import os
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print("Using: python train_word2vec.py [input_text] [output_word_vector]")
        sys.exit(1)
    input_file, output_file = sys.argv[1:3]
    sentences = []
    for line in open(input_file):
        texts = line.decode("utf-8").replace("\n", "").split("\t")[1:]
        for uter in texts:
            sentences.append(uter.split())

    model = Word2Vec(sentences, size=300, window=5, min_count=5, sg=1,
                     workers=multiprocessing.cpu_count())

    model.wv.save_word2vec_format(output_file, binary=False)
