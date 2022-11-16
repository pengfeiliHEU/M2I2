import json

train_file = '/mnt/sda/lpf/data/vqa/data_RAD/trainset.json'
test_file = '/mnt/sda/lpf/data/vqa/data_RAD/testset.json'


answer_file = '/mnt/sda/lpf/data/vqa/data_RAD/answer_list.json'

train_data_list = json.load(open(train_file, 'r'))
test_data_list = json.load(open(test_file, 'r'))
answer_list = json.load(open(answer_file, 'r'))
print(answer_list)
print(len(answer_list))

answer = []

for t in train_data_list:
    t['answer'] = str(t['answer']).lower().replace('\t', ' ')
    t['answer'] = t['answer'].replace('x ray', 'xray')
    answer.append(t['answer'])

print(answer[:5])
print(len(answer))

answer = list(set(answer))
print(answer)
print(len(answer))

