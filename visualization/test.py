import json

data = json.load(open('./trainset.json', 'r'))
print(data[:1])
print(type(data))
print(len(data))

open_data = []

for i in data:
    if i['answer_type'] == 'OPEN':
        open_data.append(i)

print(open_data[:2])
print(len(open_data))


with open('./open_data.json', "w") as f:
    json.dump(open_data, f)

# need_data = []
#
# for i in open_data:
#     if i['question'].__contains__(str(i['answer'])):
#         need_data.append(i)
#
# print(need_data)
# print(len(need_data))