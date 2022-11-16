#!/usr/bin/env bash
# Rad数据集 2min/epoch
python3 train_rad.py
# PathVQA数据集 22min/epoch
python3 train_pathvqa.py
# Slake数据集 4min/epoch
python3 train_slake.py