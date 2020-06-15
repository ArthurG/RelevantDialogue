import pandas as pd
import random

training_path = "../relevent_dialog_data_full/convai2_train_none_original_no_cands.txt"

lines = []
with open(training_path, "r") as convai2:
    lines = [line.strip() for line in convai2.readlines()]

# Build up convos 
convos = [] 
curr_convo = []
counter = 0
for i, line in enumerate(lines):
    curr = int(line.split(" ")[0])
    if curr != counter+1:
        convos.append(curr_convo)
        curr_convo = []

    counter = curr
    line = line[1:].strip().split("\t")
    curr_convo.append(line[0])
    curr_convo.append(line[1])
convos.append(curr_convo)

random.shuffle(convos)

print(len(convos))



def build_positive_examples(convos):
    ans = []
    for conv in convos:

        for i in range(len(conv)):
            for j in range(i+1, len(conv)):
                ans.append((conv[i], conv[j], 1))

    return ans

def build_negative_examples(convos, count):
    ans = []
    for i in range(len(convos)):
        for utter1 in convos[i]:
            for j in range(10):
                x = i
                while x == i:
                    x = random.randint(0, len(convos)-1)
                y = random.randint(0, len(convos[x])-1)
                ans.append((utter1, convos[x][y], 0))
    return ans

# Train test split
test = convos[:3575]
train = convos[3575:]

train_path= "../relevent_dialog_data_full/relevent_dialog_train.txt"
test_path = "../relevent_dialog_data_full/relevent_dialog_test.txt"

with open(train_path, "w") as rel_dia:
    pos = build_positive_examples(train)
    neg = build_negative_examples(train, 500)

    print(len(pos))
    print(len(neg))
    rel_dia.write("sent1\t\sent2\t\label\n")

    for x, y, z in pos:
        line = "{}\t{}\t{}\n".format(x, y, z)
        #print("line", line, "end")
        rel_dia.write(line)
    for x, y, z in neg:
        line = "{}\t{}\t{}\n".format(x, y, z)
        rel_dia.write(line)

with open(test_path, "w") as rel_dia:
    pos = build_positive_examples(test)
    neg = build_negative_examples(test, 500)

    print(len(pos))
    print(len(neg))

    rel_dia.write("sent1\t\sent2\t\label\n")
    for x, y, z in pos:
        line = "{}\t{}\t{}\n".format(x, y, z)
        #print("line", line, "end")
        rel_dia.write(line)
    for x, y, z in neg:
        line = "{}\t{}\t{}\n".format(x, y, z)
        rel_dia.write(line)
