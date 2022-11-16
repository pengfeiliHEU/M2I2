import _pickle as cPickle
import json
# 文件路径
# answer_json = '/mnt/sda/lpf/data/vqa/data_PathVQA/qas/ans2label.pkl'  # answer:index dict
# answer_json = '/mnt/sda/lpf/data/vqa/data_PathVQA/qas/q2a.pkl'  #
# answer_json = '/mnt/sda/lpf/data/vqa/data_PathVQA/qas/train_vqa.pkl'    # 训练
# answer_json = '/mnt/sda/lpf/data/vqa/data_PathVQA/qas/val_vqa.pkl'      # 验证
# answer_json = '/mnt/sda/lpf/data/vqa/data_PathVQA/qas/test_vqa.pkl'       # 测试
# answer_json = '/mnt/sda/lpf/data/vqa/data_PathVQA/qas/trainval_label2ans.pkl'  # answer list 标签
# answer_json = '/mnt/sda/lpf/data/vqa/data_PathVQA/train_img_id2idx.pkl'
# answer_list = cPickle.load(open(answer_json, 'rb'))
# print(answer_list)
# print(type(answer_list))
# print(len(answer_list))


# val_pkl = '/mnt/sda/lpf/data/vqa/data_PathVQA/qas/train_vqa.pkl'    # 训练
# val_pkl = '/mnt/sda/lpf/data/vqa/data_PathVQA/qas/val_vqa.pkl'  # 验证
val_pkl = '/mnt/sda/lpf/data/vqa/data_PathVQA/qas/test_vqa.pkl'       # 测试

val_pkl_list = cPickle.load(open(val_pkl, 'rb'))
print(val_pkl_list[:1])

val_list = []
for val in val_pkl_list:
    a_d = val['label']
    a_l = [k for k, v in a_d.items() if v == 1]
    answer = a_l[0].strip()
    # print(answer)
    v = {'qid': val['question_id'],
         'image_name': 'test/'+val['img_id']+'.jpg',
         'answer': answer,
         'answer_type': val['answer_type'],
         'question': val['sent'],
         'question_type': val['question_type']
         }
    val_list.append(v)

print(val_list[:1])
print(len(val_list))

# 保存JSON文件
# val_data_path = '/mnt/sda/lpf/data/vqa/data_PathVQA/pathvqa_test.json'
# with open(val_data_path, "w") as f:
#     json.dump(val_list, f)
######################################################################################################################
# a_test_list = []
# for val in val_pkl_list:
#     a_d = val['label']
#     a_l = [k for k, v in a_d.items() if v == 1]
#     a_test_list.append(a_l[0])

# 制作 answer list

# answer_json = '/mnt/sda/lpf/data/vqa/data_PathVQA/qas/trainval_label2ans.pkl'  # answer list 标签
# answer_list = cPickle.load(open(answer_json, 'rb'))
#
#
#
# a_test_list = list(set(a_test_list))
#
# out_num = 0
# in_num = 0
# for a in a_test_list:
#     if a in answer_list:
#         in_num += 1
#         continue
#     else:
#         out_num += 1
#         answer_list.append(a)
#         # print(a)
#
# print(out_num)
# print(in_num)
#
# print(answer_list[:10])
# print(len(answer_list))
#
# answer_path = '/mnt/sda/lpf/data/vqa/data_PathVQA/answer_all_list.json'
# with open(answer_path, "w") as f:
#     json.dump(answer_list, f)