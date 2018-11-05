import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
import random
import pickle

pos = './Data/pos.txt'
neg = './Data/neg.txt'
hm_lines = 100
global index
index = 0

def get_corpus(pos_path, neg_path):
    corpus_array = []
    corpus_final = []
    stemmer = PorterStemmer()
    with open(pos_path) as tmp:
        pos_txt = tmp.readlines()
    for line in pos_txt[:hm_lines]:
        tmp_tab = []
        tokens = word_tokenize(line.lower())
        tokens = [stemmer.stem(i) for i in tokens]
        for word in tokens:
            tmp_tab.append(word)
        corpus_array +=tmp_tab

    with open(neg_path,'r') as tmp:
        neg_txt = tmp.readlines()
    for line in neg_txt[:hm_lines]:
        tmp_tab = []
        tokens = word_tokenize(line.lower())
        tokens = [stemmer.stem(i) for i in tokens]
        for word in tokens:
            tmp_tab.append(word)
        corpus_array += tmp_tab

    cor_unique = Counter(corpus_array)
    for word in corpus_array:
        if 200 >cor_unique[word]>5:
            corpus_final.append(word)
    return (corpus_final)

def bag_of_words(file, corpus, sentiment):
     stemmer = PorterStemmer()
     vector_of_bow = []
     with open(file,'r') as tmp:
         lines = tmp.readlines()
     for txt_lines in lines[:hm_lines]:
         global index
         if index%100 == 0:
             print('index',index)
         vector = np.zeros(len(corpus))
         tokens = word_tokenize(txt_lines)
         tokens = [stemmer.stem(i) for i in tokens]
         for word in tokens:
             if word in corpus:
                 word_index = corpus.index(word)
                 vector[word_index] += 1
         vector_of_bow.append([list(vector),sentiment])
         index+=1
     vector_of_bow = list(vector_of_bow)

     return (vector_of_bow)


def train_data(pos,neg):
    corpa = get_corpus(pos, neg)
    print(len(corpa))
    # print(test_size)
    # print(type(corpa))
    final_bag = bag_of_words(pos,corpa,[0,1])
    final_bag += bag_of_words(neg,corpa,[1,0])
    test_size = int(len(final_bag)*0.2)
    random.shuffle(final_bag)
    final_bag = np.array(final_bag)
    # test = final_bag[:,1][:test_size]
    # print(type(final_bag))

    train_x = final_bag[:,0][:-test_size]
    train_y = final_bag[:,1][:-test_size]
    test_x = final_bag[:,0][-test_size:]
    test_y = final_bag[:,1][-test_size:]
    print('final bag len',len(final_bag))
    print('vector len',len(final_bag[:1,0][0]))
    # print('vector', final_bag[:1,0])
    # print('ve')
    return (train_x,train_y,test_x,test_y)


train_x, train_y, test_x, test_y = train_data(pos,neg)
# print(len(to_pickle))
dump_pickle = open('./Data/bow_data_small.pkl','wb')
pickle.dump([train_x,train_y,test_x,test_y],dump_pickle)