import json

# val_file = '/mnt/sda/lpf/data/vqa/data_Slake/test.json'
# val_file = '/mnt/sda/lpf/data/vqa/data_Slake/train.json'
val_file = '/mnt/sda/lpf/data/vqa/data_Slake/validate.json'
val_data_list = json.load(open(val_file, 'r'))
print(len(val_data_list))
print(val_data_list[:1])
print(val_data_list[-2:])
val_list = []
for val in val_data_list:
    if val['q_lang'] == 'zh':   # 过滤出英文问题
        v = {'qid': val['qid'],
             'img_id': val['img_id'],
             'image_name': val['img_name'],
             'answer': val['answer'],
             'answer_type': val['answer_type'],
             'question': val['question'],
             'question_type': val['content_type'],
             'image_organ': val['location']
             # 'content_type': val['content_type']
             }
        val_list.append(v)

print(val_list[:1])
print(len(val_list))

# 保存JSON文件
val_data_path = '/mnt/sda/lpf/data/vqa/data_Slake/zh/slake_val.json'
with open(val_data_path, "w") as f:
    json.dump(val_list, f)


# 制作标签
# train_file = '/mnt/sda/lpf/data/vqa/data_Slake/train.json'
# val_file = '/mnt/sda/lpf/data/vqa/data_Slake/validate.json'
# test_file = '/mnt/sda/lpf/data/vqa/data_Slake/test.json'
#
# train_data_list = json.load(open(train_file, 'r'))
# val_data_list = json.load(open(val_file, 'r'))
# test_data_list = json.load(open(test_file, 'r'))
#
#
# answer_trainval_list = []
# answer_all_list = []
# for t in train_data_list:
#     if t['answer'] == '':
#         continue
#     if t['q_lang'] == 'en':
#         answer_trainval_list.append(t['answer'])
#         answer_all_list.append(t['answer'])
#
# for val in val_data_list:
#     if val['answer'] == '':
#         continue
#     if val['q_lang'] == 'en':   # 过滤出英文问题
#         answer_trainval_list.append(val['answer'])
#         answer_all_list.append(val['answer'])
#
# for t in test_data_list:
#     if t['answer'] == '':
#         continue
#     if t['q_lang'] == 'en':   # 过滤出英文问题
#         answer_all_list.append(t['answer'])
# print(len(answer_trainval_list))
# print(answer_trainval_list[:10])
# answer_trainval_list = list(set(answer_trainval_list))
# print(len(answer_trainval_list))
#
# print("================")
#
# print(len(answer_all_list))
# print(answer_all_list[:10])
# answer_all_list = list(set(answer_all_list))
# print(len(answer_all_list))
#
# answer_path = '/mnt/sda/lpf/data/vqa/data_Slake/en/answer_trainval_list.json'
# with open(answer_path, "w") as f:
#     json.dump(answer_trainval_list, f)
#
# answer_all_path = '/mnt/sda/lpf/data/vqa/data_Slake/en/answer_all_list.json'
# with open(answer_all_path, "w") as f:
#     json.dump(answer_all_list, f)