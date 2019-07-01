#!/bin/bash

DATA_DIR=../../data/ubuntu_data_concat

CUDA_VISIBLE_DEVICES=0 python -u main.py \
--train_file=$DATA_DIR/train.txt \
--valid_file=$DATA_DIR/valid.txt \
--test_file=$DATA_DIR/test.txt \
--vocab_file=$DATA_DIR/vocab.txt \
--output_dir=result \
--embedding_file=../../data/embedding_w2v_d300.txt \
--maxlen_1=400 \
--maxlen_2=150 \
--hidden_size=300 \
--train_batch_size=16 \
--valid_batch_size=16 \
--test_batch_size=16 \
--fix_embedding=True \
--patience=1 \
> log.txt 2>&1 &

