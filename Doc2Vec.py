# https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python
import os.path
import numpy as np
import gensim
from gensim import corpora
from gensim.models import LsiModel, Word2Vec,doc2vec
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from gensim import similarities

def load_data(path,file_name):
    """
    Edited from website to seperate numbers from texts
    Input  : path and file_name
    Purpose: loading text file
    Output : list of paragraphs/documents and
             title(initial 100 words considred as title of document)
    """
    documents_list = []
    titles=[]
    num_list=[]
    with open( os.path.join(path, file_name) ,"r") as fin:
        for line in fin.readlines():
            num,text = line.strip().split("\t")
            documents_list.append(text)
            num_list.append(num)
    print("Total Number of Documents:",len(documents_list))
    titles.append( text[0:min(len(text),100)] )
    return documents_list,titles,num_list

def preprocess_data_Doc2Vec(doc_set):
    """
    Input  : docuemnt list
    Purpose: preprocess text (tokenize, removing stopwords, and stemming)
    Output : preprocessed text
    """
    # initialize regex tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    # create English stop words list
    en_stop = set(stopwords.words('english'))
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    # list for tokenized documents in loop
    train_corpus = []
    # loop through document list
    for i,doc in enumerate(doc_set):
        # clean and tokenize document string
        raw = doc.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [doc for doc in tokens if not doc in en_stop]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(doc) for doc in stopped_tokens]
        # tag document with id
        tagged_tokens=gensim.models.doc2vec.TaggedDocument(stemmed_tokens, [i])
        # add tokens to list
        train_corpus.append(tagged_tokens)
    return train_corpus

def load_preprocess_data(path,file_name):
    documents_list,titles,num_list=load_data(path,file_name)
    train_corpus=preprocess_data_Doc2Vec(documents_list)
    return train_corpus,documents_list,titles,num_list

train_corpus,documents_list,titles,num_list=load_preprocess_data("./","finding36279Terms.txt")
model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=1, epochs=200)
model.build_vocab(train_corpus)



def Top10matches_Doc2Vec(i,model=model,train_corpus=train_corpus):
    inferred_vector = model.infer_vector(train_corpus[i].words)
    sims = model.dv.most_similar([inferred_vector], topn=11)
    print(type(sims))
    print("="*20)
    print("For Doc2Vec\n")
    print(" Top 10 matches for %sth document: %s"%(i,documents_list[i]))
    
    count=0
    for doc_position, doc_score in sims:
        if count<=10:
            print(doc_score, documents_list[doc_position])
        count+=1
        
Top10matches_Doc2Vec(2350)
Top10matches_Doc2Vec(754)
"""
# test model
ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    second_ranks.append(sims[1])
import collections

counter = collections.Counter(ranks)
print(counter)"""