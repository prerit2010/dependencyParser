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
stack = []
relations = []
# print(train_data[0][0])

def leftArc(s1, s2):

    m, n, p, q = s1[1], s1[2], s2[1], s2[2]
    if p = -1:
        return False

    if train_data[m][n]['id'] == train_data[p][q]['head']:
        return True
    else
        return False


def rightArc(s1, s2, index_i, index_j):
    
    m, n, p, q = s1[1], s1[2], s2[1], s2[2]
    s1, s2 = train_data[m][m], train_data[p][q]
    # need to check all dependents too

    for i, word_data in enumerate(train_data[index_i]):
        w1 = (word_data['form'], index_i, i)
        w2 = (s1['form'], index_i, index_j)
        if word_data['head'] == s1['id']:
            if ((w1, w2) or (w2, w1)) not in relations
                return False

    if s1['head'] == 0:
        return True
    if s1['head'] == s2['id']:
        return True
    else
        return False


for i, data in enumerate(train_data):
    # initialize the stack with root
    stack = [("root", -1, -1),]
    # iterate over each word (acts as a buffer list)
    for j, word_data in enumerate(data):
        if j == 0:
            stack.append((word_data['form'], i, j))
        else:
            # iterate over the stack contents
            while(len(stack) > 1):
                # store top 2 elements of stack in s1 and s2
                s1, s2 = stack[-1], stack[-2]

                if leftArc(s1, s2):
                    relations.append((s2, s1))
                    stack.pop(-2)
                    operation_labels.append(0)
                elif rightArc(s1, s2, i, j):
                    relations.append((s1, s2))
                    stack.pop()
                    operation_labels.append(1)
                else:
                    stack.append((word_data['form'], i, j))
                    operation_labels.append(2)
                    break
