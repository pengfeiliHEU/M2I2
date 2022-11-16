import json
test_data_path = '/mnt/sda/lpf/data/vqa/data_PathVQA/pathvqa_test.json'
data_list = json.load(open(test_data_path, 'r'))
print(len(data_list))
print(data_list[:10])

num_num = 0
num_yn = 0
num_other = 0

for i in data_list:
    if i['answer_type'] == 'other':
        num_other += 1
    elif i['answer_type'] == 'yes/no':
        num_yn += 1
    elif i['answer_type'] == 'number':
        num_num += 1

print(num_other)
print(num_yn)
print(num_num)

print((num_other*36.28 + num_num*38.89)/(num_other+num_num))
