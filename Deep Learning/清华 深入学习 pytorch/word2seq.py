import numpy as np
import os
import re

def tokenize(content):
    content = re.sub("<.*?>", " ", content)
    content = re.sub("'s", " is", content)
    content = re.sub("'m", " am", content)
    filters = [':','\t','\n','\x97','\x96','#','$','%','&','\.']
    content = re.sub("|".join(filters), " ", content)
    tokens = [i.strip().lower() for i in content.split()]
    return tokens

class Word2Seq():
    unk_tag = 'UNK' # unknown
    pad_tag = 'PAD'
    
    unk = 0
    pad = 1
    
    def __init__(self):
        self.dict = {
            self.unk_tag: self.unk,
            self.pad_tag: self.pad
        }
        
    count = {}
    
    def fit(self, sentence): # filter words and count occurence
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1
            
    def build_vocab(self, min=5, max=None, max_features=None): # create dictionary
        # check restrictios
        if min is not None:
            self.count = {word:value for word, value in self.count.items() if value>min}
        if max is not None:
            self.count = {word:value for word, value in self.count.items() if value<max}
        if max_features is not None:
            # sort words by occurence from high to low
            temp = sorted(self.count.items(), key = lambda x:x[-1], reverse = True)[:max_features] 
            self.count = dict(temp)
        # get a dictionary {word: count}
        for word in self.count:
            self.dict[word] = len(self.dict)
        # get a inverse dictionay {index: word}
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))
    
    def transform(self, sentence, max_len=None):
        if max_len is not None:
            if max_len > len(sentence):
                sentence = sentence + [self.pad_tag]*(max_len-len(sentence)) # fill with pad
            if max_len < len(sentence):
                sentence = sentence[:max_len] # slice to max length
        return [self.dict.get(word, self.unk) for word in sentence]
    
    def inverse_transform(self, indices):
        return [self.inverse_dict.get(index) for index in indices]
    
    def __len__(self):
        return len(self.dict)
    
# # test
# ws = Word2Seq()
# ws.fit(['我','是','谁','a','b'])
# ws.fit(['我','是','我'])
# ws.build_vocab(min=0)
# print(ws.dict)
# ret = ws.transform(['我','爱','谁','a'], max_len=5)
# print(ret)
# ret = ws.inverse_transform(ret)
# print(ret)

from tqdm import tqdm

ws = Word2Seq()
data_path = './data/aclImdb/train'
temp_data_path = [os.path.join(data_path,'pos'), os.path.join(data_path,'neg')]
for path in temp_data_path:
    file_name = os.listdir(path)
    file_paths = [os.path.join(path, i) for i in file_name if i.endswith('.txt')]
    for file_path in tqdm(file_paths):
        with open(file_path,'r',encoding='UTF-8') as f:
            sentence = tokenize(f.read())
            ws.fit(sentence)

import pickle

ws.build_vocab(min=10)
pickle.dump(ws, open('./model/ws.pkl', 'wb'))
print(len(ws))