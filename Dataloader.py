import json
import os
import re


dir_path = "./Conversations/train"

for root, dirs, files in os.walk(dir_path):
    print("ok")

positive_train_data_list = []
for file in files:
    file_path = os.path.join(dir_path, file)
    with open(file_path, "r") as f:
        train_dict = json.load(f)
    # print(train_dict['history'])
    train_data_list = train_dict['history']
    i = 0
    special_string = "!,.?;"
    while i < len(train_data_list) - 1:
        # print(train_data_list[i]["uid"])
        if train_data_list[i]["uid"] != train_data_list[i+1]["uid"]:
            i = i + 1
        else:
            if train_data_list[i]["text"][-1] in special_string:
                train_data_list[i]["text"] = train_data_list[i]["text"].strip() + " " + train_data_list[i+1]["text"].strip()
            else:
                train_data_list[i]["text"] = train_data_list[i]["text"] + ". " + train_data_list[i+1]["text"].strip()
            del train_data_list[i+1]
    # print(train_data_list)

    for i, data_dict in enumerate(train_dict['history'][:-1]):
        positive_train_data = "[CLS] " + data_dict["text"].strip() + " [SEP] " + train_dict['history'][i+1]["text"]
        positive_train_data_list.append(positive_train_data)
# print(positive_train_data_list[13:15])
positive_path = './positive_train_data.txt'
with open(positive_path, "w", encoding="utf-8") as f:
    for positive_train_data in positive_train_data_list:
        if '\n' in positive_train_data:
            texts_list = positive_train_data.split("\n")
            texts = ""
            for text in texts_list:
                texts += text
            # print(texts)
            texts = texts.replace("\\", "")
            f.write(texts)
            f.write("\n")
        else:
            texts = positive_train_data.replace("\\", "")
            f.write(texts)
            f.write("\n")

with open(positive_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
# print(lines[13:15])