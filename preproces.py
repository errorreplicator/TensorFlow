import pandas as pd
import pickle
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

token_sent = []

#################################################################################
# pickle_off = open('./Data/full_txt4.pkl','rb')
# raw_txt = pickle.load(pickle_off)
# # print(type(sent))
# # print(len(sent))
# sent = sent_tokenize(raw_txt)
# index = 0
# for single_sent in sent:
#     token_sent.append(word_tokenize(single_sent))
#     if index%1000 == 0:
#         print('sentence %s out of %s'% (index,len(sent)))
#     index+=1
# print(token_sent[1:5])
# print(len(token_sent))
# pickle_on = open('./Data/token_sent.pkl','wb')
# pickle.dump(token_sent,pickle_on)
# pickle_on.close()
###################################################################################
# pickle_off = open('./Data/token_sent.pkl','rb')
# raw_sent = pickle.load(pickle_off)
# print(raw_sent[:1])
# to_remove = ['(',')',',',']','[','%',',','.','*','\'','#','$','&','_','-','+','=','@']
# stemmer = PorterStemmer()
# index = 0
# stop = set(stopwords.words('english'))
# for x in raw_sent:
#     tmp = []
#     for word in x:
#         if (word not in to_remove) and (word not in stop):
#             tmp.append(stemmer.stem(word.lower()))
#     if (index+1)%1000==0:
#         print(index+1)
#         # break
#     token_sent.append(tmp)
#     index+=1
# print(token_sent[:3])
# pickle_on = open('./Data/txt_ready.pkl','wb')
# pickle.dump(token_sent,pickle_on)
# pickle_on.close()

###################################################################################################

# pickle_open = open('./Data/txt_ready.pkl','rb')
# ready_txt = pickle.load(pickle_open)
#
# name = 'fullText_list.pkl'
# fullText_list = []
# for sent in ready_txt:
#     for word in sent:
#         fullText_list.append(word)
# print(fullText_list[:10])
# print(len(fullText_list))
# pickle_write = open('./Data/fullText_list.pkl','wb')
# pickle.dump(fullText_list,pickle_write)
# pickle_write.close()
#####################################################################################################

# remove THE and A letters
#
# open_pickle = open('./Data/txt_ready.pkl','rb')
# txt = pickle.load(open_pickle)
# txt_tab = []
# to_remove = ['--','a','the']
# for sent in txt:
#     tmp = []
#     for word in sent:
#         if word not in to_remove:
#             tmp.append(word)
#     txt_tab.append(tmp)
#
# dump_pickle = open('./Data/txt_ready.pkl','wb')
# pickle.dump(txt_tab,dump_pickle)
# dump_pickle.close()
# print(txt_tab[:3])
############################SET#############################################################

open_pickle = open('./Data/txt_ready.pkl','rb')
workTxt = pickle.load(open_pickle)

unique = []
index = 0
for sent in workTxt:
    for word in sent:
        index+=1
        if index%10000==0:
            print(index)
        unique.append(word)
print(len(unique))
wordset = set(unique)
print(len(wordset))

dump_pickle = open('./Data/txt_uniq_set.pkl','wb')
pickle.dump(wordset,dump_pickle)
dump_pickle.close()

##################################################################################

# open_pickle = open('./Data/txt_final_v3.pkl','rb')
# txt = pickle.load(open_pickle)
# tab_final = []
# to_remove = ['`']
# flag = 0
# index_error = 0
# index_glowny = 0
# for sent in txt:
#     tmp = []
#     if index_glowny % 10000 == 0:
#         print(index_glowny)
#     index_glowny += 1
#     for s in sent:
#         if s != "''" and s !='--':
#             tmp.append(s)
#     tab_final.append(tmp)
#
# print(len(tab_final))
# print(index_error)
#
# dump_pickle = open('./Data/txt_final_v4.pkl','wb')
# pickle.dump(tab_final,dump_pickle)
# dump_pickle.close()
# print('done')











