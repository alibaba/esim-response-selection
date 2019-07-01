# ESIM for Multi-turn Response Selection Task

## Introduction
If you use this code as part of any published research, please acknowledge one of the following papers.

```
@inproceedings{chen2019sequential,
  title={Sequential Matching Model for End-to-end Multi-turn Response Selection},
  author={Chen, Qian and Wang, Wen},
  booktitle={ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7350--7354},
  year={2019},
  organization={IEEE}
}
```

```
@article{DBLP:journals/corr/abs-1901-02609,
  author    = {Chen, Qian and Wang, Wen},
  title     = {Sequential Attention-based Network for Noetic End-to-End Response Selection},
  journal   = {CoRR},
  volume    = {abs/1901.02609},
  year      = {2019},
  url       = {http://arxiv.org/abs/1901.02609},
}
```

## Requirement
1. gensim
```bash
pip install gensim
```

2. Tensorflow 1.9-1.12 + Python2.7

## Steps
1. Download the [Ubuntu dataset](https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip?dl=0
) released by (Xu et al, 2017)

2. Unzip the dataset and put data directory into `data/`

3. Preprocess dataset, including concatenatate context and build vocabulary
```bash
cd data
python prepare.py
```

4. Train word2vec
```bash
bash run_train_word2vec.sh
```

5. Train and test ESIM, the log information is in `log.txt` file. You could find an example log file in `log_example.txt`.
```bash
cd scripts/esim
bash run.sh
```