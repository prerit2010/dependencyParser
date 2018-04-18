import re
from conllu import parse, parse_tree, print_tree
import time
import numpy as np
from numpy.linalg import norm
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import pandas as pd

dimension = 50
train_data = open("data/train.conllu").read()
test_data = open("data/test.conllu").read()

train_data = re.sub(r" +", r"\t", train_data)
test_data = re.sub(r" +", r"\t", test_data)

train_data = parse(train_data)
# train_data = train_data[:10000]
test_data = parse(test_data)

features = []
operation_labels = []
stack = []
relations = []
dep_relations_labels = []
# print(test_data[2 ])

def GloveVectors_Load(file_Glove):
    #Load full GloveVector file in model
    vector_file = open(file_Glove,'r')
    model = {}
    for vec in vector_file:
        splitvec = vec.split()
        word = splitvec[0]
        vector = np.array([float(value) for value in splitvec[1:]])
        model[word] = vector
    print("Done.",len(model)," words loaded!")
    return model

glove_vec = GloveVectors_Load("data/glove_data/27Bx" + str(dimension) + "d.txt")

def leftArc(s1, s2):

    m, n, p, q = s1[1], s1[2], s2[1], s2[2]
    if p == -1:
        return False

    if train_data[m][n]['id'] == train_data[p][q]['head']:
        return True
    else:
        return False


def rightArc(s1, s2, index_i, index_j):
    
    m, n, p, q = s1[1], s1[2], s2[1], s2[2]
    s1, s2 = train_data[m][n], train_data[p][q]
    
    # for all words in the current sentence
    for i, word_data in enumerate(train_data[index_i]):
        w1 = (word_data['lemma'], index_i, i)
        w2 = (s1['lemma'], index_i, index_j)
        # if the word is dependent of s1
        if word_data['head'] == s1['id']:
            # if its relation has not been processed with s1
            if (w1, w2)  not in relations:
                # cannot remove s1 from stack, return false.
                return False

    if s1['head'] == 0:
        return True
    if s1['head'] == s2['id']:
        return True
    else:
        return False

vocab = set()
pos_tags = set()
deprel = set()
for data in train_data:
    for word_data in data:
        pos_tags.add(word_data['xpostag'])
        deprel.add(word_data['deprel'])
vocab.add("root")


pos_tags_train =  {tag:i for i, tag in enumerate(pos_tags)}
dep_rel_train =  {dep:i  for i, dep in enumerate(deprel)}
# print(pos_tags)

start_time = time.time()
for i, data in enumerate(train_data):
    # initialize the stack with root
    if i%200 == 0:
        print("Processed document : %d" % i)
    stack = [("root", -1, -1),]
    rel = []
    # iterate over each word (acts as a buffer list)
    for j, word_data in enumerate(data):
        # print("processed word : %d" % j)
        if j == 0:
            stack.append((word_data['lemma'], i, j))
        else:
            # iterate over the stack contents
            while(len(stack) > 1):
                # store top 2 elements of stack in s1 and s2
                s1, s2 = stack[-1], stack[-2]
                stack_vec = glove_vec.get(s1[0])
                sig0_word = np.zeros((dimension)) if stack_vec is None else stack_vec
               
                sig0_pos = pos_tags_train[train_data[s1[1]][s1[2]]['xpostag']]
                sig0_deprel = dep_rel_train[train_data[s1[1]][s1[2]]['deprel']]
                beta0_vec = glove_vec.get(word_data['lemma'])
                beta0_word = np.zeros((dimension)) if beta0_vec is None else beta0_vec
                # beta0_word = vocab_train[word_data['lemma']]
                beta0_pos = pos_tags_train[word_data['xpostag']]
                beta0_deprel = dep_rel_train[word_data['deprel']]
                beta1_word = np.zeros((dimension))
                try:
                    beta1_word = glove_vec.get(train_data[i][j+1]['lemma'])
                    beta1_word = np.zeros((dimension)) if beta1_word is None else beta1_word
                except:
                    beta1_word = np.zeros((dimension))
                
                beta1_pos = -1
                try:
                    beta1_pos = pos_tags_train[train_data[i][j+1]['xpostag']]
                except:
                    beta1_pos = -1

                beta1_deprel = 0
                try:
                    beta1_deprel = dep_rel_train[train_data[i][j+1]['deprel']]
                except:
                    beta1_deprel = -1

                feat = np.array([sig0_deprel, beta0_deprel, sig0_pos, beta0_pos, beta1_pos,\
                          beta1_deprel])
                # print(feat.shape)
                # print(sig0_word.shape, beta0_word.shape, beta1_word.shape)
                feat = np.concatenate((feat, sig0_word, beta0_word, beta1_word))
                features.append(feat)

                if leftArc(s1, s2):
                    rel.append((s2, s1))
                    stack.pop(-2)
                    s2_deprel = dep_rel_train[train_data[s2[1]][s2[2]]['deprel']]
                    dep_relations_labels.append(s2_deprel)
                    operation_labels.append(0)
                    # print("left")
                elif rightArc(s1, s2, i, j):
                    rel.append((s1, s2))
                    stack.pop()
                    s1_deprel = dep_rel_train[train_data[s1[1]][s1[2]]['deprel']]
                    dep_relations_labels.append(s1_deprel)
                    operation_labels.append(1)
                    # print("right")
                else:
                    stack.append((word_data['lemma'], i, j))
                    dep_relations_labels.append(-1)
                    operation_labels.append(2)
                    # print("shift")
                    break
                

    # relations.append(rel)

    while(len(stack) > 1):

        s1 = stack[-1]
        for st in stack:
            s2 = {}
            if st == ("root", -1, -1):
                s2 = {'id' : 0}
            else:
                s2 = train_data[st[1]][st[2]]
            s1_head = train_data[s1[1]][s1[2]]['head']
            if s2['id'] == s1_head:
                rel.append((s1, st))
                operation_labels.append(1)

        # stack[-2]
                stack_vec = glove_vec.get(s1[0])
                sig0_word = np.zeros((dimension)) if stack_vec is None else stack_vec
        
                sig0_pos = pos_tags_train[train_data[s1[1]][s1[2]]['xpostag']]
                sig0_deprel = dep_rel_train[train_data[s1[1]][s1[2]]['deprel']]
        
                beta0_word = np.zeros((dimension)) #if beta0_vec is None else beta0_vec
                # beta0_word = vocab_train[word_data['lemma']]
                beta0_pos = -1 #pos_tags_train[word_data['xpostag']]
                beta0_deprel = -1 #dep_rel_train[word_data['deprel']]
                beta1_word = np.zeros((dimension))
            

                beta1_pos = -1

                beta1_deprel = -1

                feat = np.array([sig0_deprel, beta0_deprel, sig0_pos, beta0_pos, beta1_pos,\
                          beta1_deprel])
                
                feat = np.concatenate((feat, sig0_word, beta0_word, beta1_word))
                features.append(feat)
                s1_deprel = dep_rel_train[train_data[s1[1]][s1[2]]['deprel']]
                dep_relations_labels.append(s1_deprel)

        stack.pop()
        
    relations.append(rel)

X_train = features
Y_train = operation_labels
Y_train_dep = dep_relations_labels
# Y_train = np.array(pd.get_dummies(Y_train))

features = []
operation_labels = []
stack = []
relations = []
dep_relations_labels = []

train_data = test_data

vocab = set()
pos_tags = set()
deprel = set()
for data in train_data:
    for word_data in data:
        # vocab.add(word_data['lemma'])
        pos_tags.add(word_data['xpostag'])
        deprel.add(word_data['deprel'])
vocab.add("root")

pos_tags_train =  {tag:i for i, tag in enumerate(pos_tags)}
dep_rel_train =  {dep:i  for i, dep in enumerate(deprel)}
# print(pos_tags)

start_time = time.time()
for i, data in enumerate(train_data):
    # initialize the stack with root
    if i%200 == 0:
        print("Processed document : %d" % i)
    stack = [("root", -1, -1),]
    rel = []
    # iterate over each word (acts as a buffer list)
    for j, word_data in enumerate(data):
        # print("processed word : %d" % j)
        if j == 0:
            stack.append((word_data['lemma'], i, j))
        else:
            # iterate over the stack contents
            while(len(stack) > 1):
                # store top 2 elements of stack in s1 and s2
                s1, s2 = stack[-1], stack[-2]
                stack_vec = glove_vec.get(s1[0])
                sig0_word = np.zeros((dimension)) if stack_vec is None else stack_vec
                
                sig0_pos = pos_tags_train[train_data[s1[1]][s1[2]]['xpostag']]
                sig0_deprel = dep_rel_train[train_data[s1[1]][s1[2]]['deprel']]
                beta0_vec = glove_vec.get(word_data['lemma'])
                beta0_word = np.zeros((dimension)) if beta0_vec is None else beta0_vec
                # beta0_word = vocab_train[word_data['lemma']]
                beta0_pos = pos_tags_train[word_data['xpostag']]
                beta0_deprel = dep_rel_train[word_data['deprel']]
                beta1_word = np.zeros((dimension))
                try:
                    beta1_word = glove_vec.get(train_data[i][j+1]['lemma'])
                    beta1_word = np.zeros((dimension)) if beta1_word is None else beta1_word
                except:
                    beta1_word = np.zeros((dimension))
                    # print("ok2")

                
                beta1_pos = -1
                try:
                    beta1_pos = pos_tags_train[train_data[i][j+1]['xpostag']]
                except:
                    beta1_pos = -1

                beta1_deprel = 0
                try:
                    beta1_deprel = dep_rel_train[train_data[i][j+1]['deprel']]
                except:
                    beta1_deprel = -1

                feat = np.array([sig0_deprel, beta0_deprel, sig0_pos, beta0_pos, beta1_pos,\
                          beta1_deprel])
                
                feat = np.concatenate((feat, sig0_word, beta0_word, beta1_word))
                features.append(feat)

                if leftArc(s1, s2):
                    rel.append((s2, s1))
                    stack.pop(-2)
                    s2_deprel = dep_rel_train[train_data[s2[1]][s2[2]]['deprel']]
                    dep_relations_labels.append(s2_deprel)
                    operation_labels.append(0)
                    # print("left")
                elif rightArc(s1, s2, i, j):
                    rel.append((s1, s2))
                    stack.pop()
                    s1_deprel = dep_rel_train[train_data[s1[1]][s1[2]]['deprel']]
                    dep_relations_labels.append(s1_deprel)
                    operation_labels.append(1)
                    
                else:
                    stack.append((word_data['lemma'], i, j))
                    dep_relations_labels.append(-1)
                    operation_labels.append(2)
                    
                    break

    while(len(stack) > 1):

        s1 = stack[-1]
        for st in stack:
            s2 = {}
            if st == ("root", -1, -1):
                s2 = {'id' : 0}
            else:
                s2 = train_data[st[1]][st[2]]
            s1_head = train_data[s1[1]][s1[2]]['head']
            if s2['id'] == s1_head:
                rel.append((s1, st))
                operation_labels.append(1)

        
                stack_vec = glove_vec.get(s1[0])
                sig0_word = np.zeros((dimension)) if stack_vec is None else stack_vec
        
                sig0_pos = pos_tags_train[train_data[s1[1]][s1[2]]['xpostag']]
                sig0_deprel = dep_rel_train[train_data[s1[1]][s1[2]]['deprel']]
        
                beta0_word = np.zeros((dimension)) #if beta0_vec is None else beta0_vec
                
                beta0_pos = -1 #pos_tags_train[word_data['xpostag']]
                beta0_deprel = -1 #dep_rel_train[word_data['deprel']]
                beta1_word = np.zeros((dimension))
           
        
                beta1_pos = -1

                beta1_deprel = -1

                feat = np.array([sig0_deprel, beta0_deprel, sig0_pos, beta0_pos, beta1_pos,\
                          beta1_deprel])
                
                feat = np.concatenate((feat, sig0_word, beta0_word, beta1_word))
                features.append(feat)
                s1_deprel = dep_rel_train[train_data[s1[1]][s1[2]]['deprel']]
                dep_relations_labels.append(s1_deprel)

        stack.pop()
    relations.append(rel)  
 

X_test = features
Y_test = operation_labels
Y_test_dep = dep_relations_labels


print("\nTotal processing time : ", time.time()-start_time)

print("Training...")
# clf = MultinomialNB()
# clf = LogisticRegression()
# clf = svm.SVC(verbose=True)
# clf = MLPClassifier(learning_rate_init=0.001, learning_rate='adaptive', verbose=True, max_iter=500, hidden_layer_sizes=5)
clf = MLPClassifier(learning_rate_init=0.001, verbose=True, max_iter=50, hidden_layer_sizes=200)
start_time = time.time()
clf.fit(X_train, Y_train)
print("Training completed in %d Seconds" % int(time.time()-start_time))

start_time = time.time()
Y_pred = clf.predict(X_test)
print("Testing completed in %d Seconds" % int(time.time()-start_time))
print(accuracy_score(Y_test, Y_pred))

print("Training for deprel...")
# clf = MultinomialNB()
# clf = LogisticRegression()
# clf = svm.SVC(verbose=True)
# clf = MLPClassifier(learning_rate_init=0.001, learning_rate='adaptive', verbose=True, max_iter=500, hidden_layer_sizes=5)
clf = MLPClassifier(learning_rate_init=0.001, verbose=True, max_iter=50, hidden_layer_sizes=5)
start_time = time.time()
clf.fit(X_train, Y_train_dep)
print("Training completed in %d Seconds" % int(time.time()-start_time))

start_time = time.time()
Y_pred = clf.predict(X_test)
print("Testing completed in %d Seconds" % int(time.time()-start_time))
print(accuracy_score(Y_test_dep, Y_pred))
