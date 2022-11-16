import json
import pandas as pd
import pyarrow as pa
import gc
import random
import os
import csv

from tqdm import tqdm
from glob import glob


# 将csv文件转换成json文件
def csv2json(root, dataset_root):
    for split in ["train", "valid"]:  # 83275  7645

        with open(f"{root}/ImageCLEFmedCaption_2022_caption_prediction_{split}.csv", "r") as fp:
            reader = csv.reader(fp)
            header = next(reader)
            print(header)
            print(header[0].split('\t'))

            all_items = []
            count = 0
            max_len = 0  # 最长的句子分别是410和297
            len_count = 0  # 平均长度分别是19.5和21.5
            for item in tqdm(reader):
                iid2caption = {}
                item = item[0].split("\t")  # [filename, caption]
                iid = "%s/%s%s" % (split, item[0], ".jpg")

                # 统计caption长度
                # word_list = item[1].split(' ')
                # len_count += len(word_list)
                # max_len = max(max_len, len(word_list))

                iid2caption['image'] = iid
                iid2caption['caption'] = item[1]
                all_items.append(iid2caption)
                count += 1
                # print(iid2caption)

            # 保存到json文件
            result_file = os.path.join(dataset_root, 'icmc_2022_%s.json' % split)
            json.dump(all_items, open(result_file, 'w'))

            print("count: ", count)
            # print("max_len: ", max_len)
            # print("avg_len: ", len_count / count)


if __name__ == "__main__":
    root = "/home/hoo/CVdataset/Medical_Dataset/ImageCLEFmed-Caption-2022"
    csv2json(root=root, dataset_root="../data/")
