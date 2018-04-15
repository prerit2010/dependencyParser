import re
from conllu import parse, parse_tree, print_tree

train_data = open("data/train.conllu").read()
test_data = open("data/test.conllu").read()

train_data = re.sub(r" +", r"\t", train_data)
test_data = re.sub(r" +", r"\t", test_data)

train_data = parse(train_data)
test_data = parse(test_data)

configurations = []
operation_labels = []
stack = [("root", -1, -1),]
relations = []
print(train_data[0][0])

def leftArc(s1, s2):

    m, n, p, q = s1[1], s1[2], s2[1], s2[2]
    if p = -1:
        return False

    if train_data[m][n]['id'] == train_data[p][q]['head']:
        return True
    else
        return False

def rightArc(s1, s2):
    
    m, n, p, q = s1[1], s1[2], s2[1], s2[2]
    # need to check all dependents too
    if train_data[m][n]['head'] == 0:
        return True
    if train_data[m][n]['head'] == train_data[p][q]['id']:
        return True
    else
        return False

for i, data in enumerate(train_data):
    for j, word_data in enumerate(data):
        if j == 0:
            stack.append((word_data['form'], i, j))
        else:
            while(1):
                if len(stack) < 2:
                    break
                s1 = stack[-1]
                s2 = stack[-2]

                if leftArc(s1, s2):
                    relations.append((s2, s1))
                    stack.pop(-2)
                    operation_labels.append(0)
                elif rightArc(s1, s2):
                    relations.append((s1, s2))
                    stack.pop()
                    operation_labels.append(1)
                else:
                    stack.append((word_data['form'], i, j))
                    operation_labels.append(2)
                    break
