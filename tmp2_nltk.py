from nltk.corpus import gutenberg
# from nltk.book import gutenberg
# from nltk.corpus import semcor
import nltk
# from basic import loadData
from nltk.tag import pos_tag_sents,pos_tag
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
# import timer

print(gutenberg.fileids())
print(gutenberg.words('whitman-leaves.txt'))

# FILE LOAD DATA ##########################

with open('Data/VariedTypes.txt','r') as reader:
    rawTxt = reader.read().replace('\n',' ')

newText = rawTxt.replace('  ','')
tabTxt = newText.split(' ')
# print(tabTxt)
txt113to222 = tabTxt[114:222]
print(txt113to222)
exitTxt = ' '.join(txt113to222)
print(exitTxt)

tags = [word_tokenize(exitTxt)]
print(tags)
partofsp = pos_tag_sents(tags) # part of speach tagging

print(partofsp)


#######################################################


myString = "This is the text I want to tag. Hope this is good. Well, if it's not than I need to figure out something else."

# word and sentence  TOKENIZATION

tokens = word_tokenize(myString) # word tokenization #len 29
sent = sent_tokenize(myString) #sentence tokenization # len 3
print('tokens ', tokens)
print('sent ',sent)
# print(len(tokens))
# print(len(sent))

############################# STOPWORDS #################################

stop_words = set(stopwords.words('english'))

myTokensStop = [i for i in tokens if not i in stop_words]
# print(myTokensStop)

myTokensStop = []

for i in tokens:
    if i not in stop_words:
        myTokensStop.append(i)

print('Tokens no stop-words ',myTokensStop)

##########################################################################
###########POS Part Of Speach Tagging || WORDS and SENTENCES##############
# pos_tokens = tokens
pos_words = pos_tag(tokens)
print('pos_tag function ',pos_words)

simplified_pos = pos_tag(tokens, tagset="universal")
print('pos_tag with universal paramiter: ',simplified_pos) # simplifiend part of speach tagging

sentences = [word_tokenize(i) for i in sent]
print(sentences)
pos_sent = pos_tag_sents(sentences)
all_pos = pos_tag_sents(sentences, tagset='universal') #, tagset="universal")

print('this is pos_tag_sent normal: ' + str(pos_sent))
print('this is pos_tag_sent with universal option: ' + str(all_pos))

# tagged_sents = nltk.pos_tag_sents(tokenized_sentences)

#####################Finding POS in tagged text#############################

nnp = []
for i in range(0,len(pos_sent)):
    if pos_sent[i][1]== 'NNP':
        nnp.append(pos_sent[i][0])
print(len(nnp))
print(nnp)

########################POS help############################################

print(nltk.help.upenn_tagset('NNP'))

##########################Lematizing###################################
from nltk.stem import WordNetLemmatizer
print(tokens)
tab_lema = []
lematizer = WordNetLemmatizer()
for i in tokens:
    tab_lema.append(lematizer.lemmatize(i))
print(tab_lema)

##########################Stemming#####################################
from nltk.stem import PorterStemmer
tab_stem = []
stemmer = PorterStemmer()

for i in tokens:
    tab_stem.append(stemmer.stem(i))
print(tab_stem)

######################################################################
############################NAME ENTITY RECOGNITION###################
from nltk import pos_tag, word_tokenize, ne_chunk

rawTxt = 'This is New York and Alabama together working for John. Oracle is not quite yet there.'
tokens = word_tokenize(rawTxt)

pos_tokens = pos_tag(tokens)
ne_tokens = ne_chunk(pos_tokens)
# ne_tokens = ne_chunk(pos_tokens, binary=True) # only tag YES / NO [NE] if Name entity no distigtion for type
print(ne_tokens) #type <class 'nltk.tree.Tree'>

#################################################################################################################
######################################################################
#########################################

from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

token_sent = []
clean_data = []
stemmer = PorterStemmer()

File = '../../Dataset/Gutenberg/txt/William Wordsworth___The Prose Works of William Wordsworth.txt'
with open(File,'r') as tmp:
    intTxt = tmp.read().replace('\n',' ')

stop = set(stopwords.words('english'))
to_remove = ['(',')',',',']','[','%',',','.','*','\'']
# intTxt = intTxt.lower()
# rawTxt = intTxt.split()
sentTxt = sent_tokenize(intTxt)
# print(len(sentTxt))
# print(sentTxt[20:22])
for x in sentTxt:
    token_sent.append(word_tokenize(x))

# print(token_sent[50:53])

for sent in token_sent:
    tmp = []
    for word in sent:
        flag = 0
        for y in word:
            if y in to_remove:
                flag = 1
        if flag !=1 and word not in stop:
            tmp.append(stemmer.stem(word.lower()))
    clean_data.append(tmp)

# print(clean_data[50:53])

######### just test ##############
#stop words, stema
model_old = Word2Vec.load('Data/first_model.w2v')
model_new = Word2Vec(sentences=clean_data, size=64, sg=1, window=10, min_count=5, seed=42, workers=4)

word = 'woman'
print(model_old.most_similar(word))
print(model_new.most_similar(word))