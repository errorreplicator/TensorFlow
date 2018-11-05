import os, glob
import pickle
path = 'C:/Dataset/Gutenberg/txt/subset/'
filenames = []
rawText = ''
fullText = ''
index = 0
for file in glob.glob(os.path.join(path,'*.txt')):
    # print(path)
    # filenames.append(file)
    # if file.split('\\')[1].startswith('Char'):
    with open(file,'r',encoding='iso-8859-1') as tmp:
        rawText = tmp.read().replace('\n','')
        filenames.append(file)

    fullText += rawText
    print(index, len(fullText))
    index+=1
    # if index>500:
    #     break
# pd.to_pickle(fullText,'./Data/fullText.pkl')
# print(len(rawText))
print(len(filenames))

dump_pickle = open('./Data/full_txt4.pkl','wb')
pickle.dump(fullText,dump_pickle)
dump_pickle.close()


