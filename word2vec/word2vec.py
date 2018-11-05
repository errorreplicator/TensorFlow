import pickle
import pandas as pd
import numpy as np
# ./Data/full_txt4.pkl -> raw string lenght #count 19 589 065 #type string
# txt_ready.pkl -> sentences with tokens - no punktuation, no stop words, stemma # count 1 936 668
# ./Data/txt_uniq_set.pkl -> unique set # count 160 410
# word_pairs.pkl -> lista pary slow

# open_pickle = open('./Data/txt_ready.pkl','rb')
# sentences = pickle.load(open_pickle)
#
# print(sentences[:3])


open_set = open('./Data/txt_uniq_set.pkl','rb')
txt_set = pickle.load(open_set)

# print (len(txt_set))
VOCAB_SIZE = len(txt_set)
# print(type(txt_set))
index2wor = {}
words2ind = {}
for ind,wor in enumerate(txt_set):
    index2wor[ind] = wor
    words2ind[wor] = ind

# WINDOW_SIZE = 2
#
# data = []
# for sentence in sentences:
#     for word_index, word in enumerate(sentence):
#         for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :
#             if nb_word != word:
#                 data.append([word, nb_word])

# dump_pickle = open('./Data/word_pairs.pkl','wb')
# pickle.dump(data,dump_pickle)
# dump_pickle.close()

open_pairs = open('./Data/word_pairs.pkl','rb')
data = pickle.load(open_pairs)
print(VOCAB_SIZE)
print(len(data))
def one_hot(data_point_index, vocab_size):
    tmp = np.zeros(vocab_size)
    tmp[data_point_index] = 1
    return tmp


# print(data)
x_train = []
y_train = []
print(len(data))
index = 0
try :
    for pairs in data:
        x_train.append(one_hot(words2ind[pairs[0]], VOCAB_SIZE))
        y_train.append(one_hot(words2ind[pairs[1]], VOCAB_SIZE))
        if index%100==0:
            print(index,'out of: ',len(data))
        index+=1
        if index%5000==0:
            break
except Exception as inst:
    print(type(inst))    # the exception instance
    print(inst.args)     # arguments stored in .args
    print(inst)

dump_pickle = open('./Data/one_hot_table.pkl','wb')
pickle.dump([x_train,y_train],dump_pickle)
dump_pickle.close()


# print(data[:10])
# print(words2ind)




