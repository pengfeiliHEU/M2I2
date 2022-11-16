import sys
import json
import random
import os
import argparse
import yaml

from vqaTools.vqa import *
from vqaTools.vqaEval import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quesFile', default='/mnt/sda/lpf/data/vqa/data_PathVQA/pathvqa_test.json')
    parser.add_argument('--resFile', default='./output/D-pathvqa/result/med_pretrain_29_vqa_result_<epoch>.json')
    # parser.add_argument('--resFile', default='./output/pathvqa/result/_vqa_result_<epoch>.json')
    args = parser.parse_args()

    all_result_list = []

    quesFile = args.quesFile
    vqa = VQA(quesFile, quesFile)  # question 和 answer 和 imgToQA
    for i in range(40):
        resFile = args.resFile.replace('<epoch>', str(i))
        print(resFile)

        # create vqa object and vqaRes object
        # 两个都是VQA对象实例，数据处理够保存在imgToQA、qa和qqa成员变量中
        vqaRes = vqa.loadRes(resFile, quesFile)  # question 和 predict answer 和 imgToQA

        # create vqaEval object by taking vqa and vqaRes
        vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2
        # evaluate results
        vqaEval.evaluate()

        # print accuracies
        acc_dict = {}
        print("\n")
        print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
        acc_dict['Epoch'] = i + 1
        acc_dict['Overall'] = vqaEval.accuracy['overall']
        print("Per Answer Type Accuracy is the following:")
        for ansType in vqaEval.accuracy['perAnswerType']:
            acc_dict[ansType] = vqaEval.accuracy['perAnswerType'][ansType]
            print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
        print("\n")

        # save evaluation results to ./results folder
        accuracyFile = resFile.replace('.json', '_acc.json')
        json.dump(vqaEval.accuracy, open(accuracyFile, 'w'))
        compareFile = resFile.replace('.json', '_compare.json')
        json.dump(vqaEval.ansComp, open(compareFile, 'w'))

        all_result_list.append(acc_dict)
    index = args.resFile.rfind('/')
    compareFile = args.resFile[0:index]
    compareFile = os.path.join(compareFile, 'all_acc.json')
    json.dump(all_result_list, open(compareFile, 'w'))
    print('All accurary file saved to: ', compareFile)
