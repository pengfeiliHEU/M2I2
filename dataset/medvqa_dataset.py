import os
import re
import json
import random
import argparse
from PIL import Image
from torch.utils.data import Dataset
# from dataset.utils import pre_question
# from pathlib import Path
# from torchvision import transforms
# import ruamel_yaml as yaml
import yaml
import _pickle as cPickle


contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
    "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd",
    "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm",
    "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
    "someone'dve": "someone'd've", "someonell": "someone'll", "someones": "someone's",
    "somethingd": "something'd", "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're",
    "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
    "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd",
    "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll",
    "youre": "you're", "youve": "you've"
}

manual_map = {'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
              'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
# re.compile 编译一个正则表达式模式，返回一个 Pattern 对象。
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
         '(', ')', '=', '+', '\\', '_', '-',
         '>', '<', '@', '`', ',', '?', '!']


# Notice that VQA score is the average of 10 choose 9 candidate answers cases
# See http://visualqa.org/evaluation.html
def get_score(occurences):
    if occurences == 0:
        return .0
    elif occurences == 1:
        return .3
    elif occurences == 2:
        return .6
    elif occurences == 3:
        return .9
    else:
        return 1.


# 处理标点符号，去除标点符号
def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


# 1、将英文数字转成阿拉伯数字；
# 2、转成小写；
# 3、去掉a、an、the；
# 4、将压缩的word替换成对应的形式，如dont -> don't
def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


# 多次替换
def multiple_replace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text


# answer的预处理
def preprocess_answer(answer):
    answer = str(answer)
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '').replace('x ray', 'xray')
    return answer


def pre_question(question: str, max_ques_words: int):
    # 转成小写
    question = question.lower()

    # 将question中无关提示字符去除
    if "? -yes/no" in question:
        question = question.replace("? -yes/no", "")
    if "? -open" in question:
        question = question.replace("? -open", "")
    if "? - open" in question:
        sentence = question.replace("? - open", "")
    # 将一些符号去除
    question = question.replace(',', '').replace('?', '').replace('\'s',
                ' \'s').replace('...', '').replace('x ray', 'x-ray').replace('.', '')

    # 去除字符串右边的空格
    question = question.rstrip(' ')
    # truncate question
    question_words = question.split(' ')
    if len(question_words) > max_ques_words:
        question = ' '.join(question_words[:max_ques_words])

    return question


class rad_dataset(Dataset):
    def __init__(self, ann_file, transform, rad_root, eos='[SEP]', split="train", max_ques_words=30,
                 answer_list=''):
        self.split = split
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))

        self.transform = transform
        self.rad_root = rad_root
        self.max_ques_words = max_ques_words
        self.eos = eos

        if split == 'test':
            self.max_ques_words = 50  # do not limit question length during test
            self.answer_list = json.load(open(answer_list, 'r'))
            # self.answer_list = cPickle.load(open(answer_list, 'rb'))
            # print(self.answer_list)
            # print(len(self.answer_list))


    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.rad_root, ann['image_name'])

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        if self.split == 'test':
            question = pre_question(ann['question'], self.max_ques_words)
            question_id = ann['qid']
            return image, question, question_id
        elif self.split == 'train':
            question = pre_question(ann['question'], self.max_ques_words)

            answers = ann['answer']
            answers = [preprocess_answer(answers)]
            weights = [0.5]

            answers = [answer + self.eos for answer in answers]

            return image, question, answers, weights

class pathvqa_dataset(Dataset):
    def __init__(self, ann_file, transform, rad_root, eos='[SEP]', split="train", max_ques_words=30,
                 answer_list=''):
        self.split = split
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))

        self.transform = transform
        self.rad_root = rad_root
        self.max_ques_words = max_ques_words
        self.eos = eos

        if split == 'test':
            self.max_ques_words = 50  # do not limit question length during test
            self.answer_list = json.load(open(answer_list, 'r'))
            # self.answer_list = cPickle.load(open(answer_list, 'rb'))
            # print(self.answer_list)
            # print(len(self.answer_list))


    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.rad_root, ann['image_name'])

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        if self.split == 'test':
            question = pre_question(ann['question'], self.max_ques_words)
            question_id = ann['qid']
            return image, question, question_id
        elif self.split == 'train':
            question = pre_question(ann['question'], self.max_ques_words)

            answers = ann['answer']
            answers = [preprocess_answer(answers)]
            weights = [0.5]

            answers = [answer + self.eos for answer in answers]

            return image, question, answers, weights

class slake_dataset(Dataset):
    def __init__(self, ann_file, transform, rad_root, eos='[SEP]', split="train", max_ques_words=30,
                 answer_list=''):
        self.split = split
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))

        self.transform = transform
        self.rad_root = rad_root
        self.max_ques_words = max_ques_words
        self.eos = eos

        if split == 'test':
            self.max_ques_words = 50  # do not limit question length during test
            self.answer_list = json.load(open(answer_list, 'r'))


    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.rad_root, ann['image_name'])

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        if self.split == 'test':
            question = pre_question(ann['question'], self.max_ques_words)
            question_id = ann['qid']
            return image, question, question_id
        elif self.split == 'train':
            question = pre_question(ann['question'], self.max_ques_words)

            answers = ann['answer']
            answers = [preprocess_answer(answers)]
            weights = [0.5]

            answers = [answer + self.eos for answer in answers]

            return image, question, answers, weights


