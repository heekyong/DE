import os
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


def get_relative_path(path):
    filenames = []
    filepaths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root,file)
            filepaths.append(filepath)
            filenames.append(file)
    return filenames,filepaths


def read_urls(path):
    filenames = get_relative_path(path)
    url_lists = []
    for filename in filenames:
        urls = open(filename).readlines()
        for url in urls:
            url = url.strip()
            url_lists.append(url)
    return url_lists


def read_docs(path):
    filenames,filepaths = get_relative_path(path)
    docs = []
    doc_names = []
    stopwrds = stopwords.words('english')
    porter_stemmer = PorterStemmer()
    for filepath,filename in zip(filepaths,filenames):
        f = open(filepath, 'r', encoding='utf-8')
        doc = ''
        temp =  f.readlines()
        for line in temp:
            doc += line.lower()
        doc = nltk.word_tokenize(doc)
        doc = [token for token in doc if token not in stopwrds]
        docs.append(doc)
        doc_names.append(filename)
    return doc_names, docs


def index_doc(docs,doc_names):
    index = {}
    inverse_index = {}
    for doc,doc_name in zip(docs,doc_names):
        word_count={}
        for word in doc:
            if word in word_count.keys():
                word_count[word]+=1
            else:
                word_count[word]=1
        index[doc_name] = word_count
        #index[doc_name]=nltk.Text(doc).vocab()
    for doc in index.keys():
        doc_index = index[doc]
        for word in doc_index.keys():
            if word in inverse_index.keys():
                inverse_index[word].append(doc)
            else:
                inverse_index[word] = [doc]
    return index, inverse_index


def build_dictionary(index):
    dictionary = {}
    for word in index.keys():
        dictionary[word]=len(dictionary)
    return dictionary


def compute_tfidf(index,word_dictionary,doc_dictionary):
    vocab_size = len(word_dictionary)
    doc_size = len(doc_dictionary)
    tf = np.zeros((doc_size,vocab_size))
    for doc in index:
        index_per_doc = index[doc]
        vector = np.zeros(vocab_size)
        for word in index_per_doc:
            vector[word_dictionary[word]] = index_per_doc[word]
        vector = np.log(vector+1)
        tf[doc_dictionary[doc]] = vector
    idf_numerator = doc_size
    idf_denominator = np.sum(np.sign(tf),0)
    idf = np.log(idf_numerator/idf_denominator)
    tfidf = tf*idf
    return tfidf


def cosine_similarity(x,y):
    normalizing_factor_x = np.sqrt(np.sum(np.square(x)))
    normalizing_factor_y = np.sqrt(np.sum(np.square(y)))
    return np.matmul(x,np.transpose(y))/(normalizing_factor_x*normalizing_factor_y)


def query_matching(inverse_dictionary,query):
    set_list = [set(inverse_dictionary[word]) for word in query]
    return set.union(*set_list)


if __name__ == '__main__':
    doc_names, docs = read_docs('data')
    index, inverted_index = index_doc(docs,doc_names)
    word_dictionary = build_dictionary(inverted_index)
    doc_dictionary = build_dictionary(index)
    tfidf = compute_tfidf(index,word_dictionary,doc_dictionary)


f = open('input_document.txt','r')
query = f.readline()
print(query)
f.close()
tokens = nltk.word_tokenize(query)
tokens
stop = set(stopwords.words('english'))
tokens = [ i for i in tokens if i not in stop]
query=tokens

new={}
for token in tokens:
    if token in inverted_index.keys():
        new[token]=inverted_index[token]
new
new_list=[]
for i in new:
    print(new[i])
    new_list.extend(new[i])
new_list = list(set(new_list))
new_list.sort()
new_list #인버티드인덱스 사용해서 인풋다큐먼트에 있는 단어(스탑워드빼고)를 하나라도 가지는 데이터셋만 뉴리스트에 새로 저장

"""
#new_list에 있는 데이터파일들만 data_new폴더에 해당 텍스트들만 옮겨서 다시
doc_names, docs = read_docs('data_new')
index, inverted_index = index_doc(docs,doc_names)
word_dictionary = build_dictionary(inverted_index)
doc_dictionary = build_dictionary(index)
tfidf = compute_tfidf(index,word_dictionary,doc_dictionary)
"""

for word in word_dictionary:
    if word == 'bruno':
        print(word_dictionary[word])
    if word == 'mars':
        print(word_dictionary[word])
    if word == 'back':
        print(word_dictionary[word])
    else :
        pass
#tfidf matrix에서 쿼리에 해당하는 열 번호 찾기
# => 각각 열번호 265,266,302임.
score=[]

for doc in doc_dictionary.values():
    score.append(tfidf[doc,265]+tfidf[doc,266]+tfidf[doc,302])
    print('score of the doc %d is %f' %(doc, score[doc]))
# documnet 별 각각의 score 계산하여 출력

mx = 0.0
index=0

for num in score:
    if num > mx:
        mx=num
        index=score.index(num)
    else:
        pass
print('가장높은 스코어는 %f점인 doc %d이다.' %(mx,index))

# 결과 출력시 "가장높은 스코어는 4.422939점인 doc 1이다."가 나온다.
# index1을 가지는 문서는 entertainment01.txt이다. 따라서 input query의 문서는 entertainment로 label한다.
