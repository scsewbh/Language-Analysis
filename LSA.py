# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 20:12:06 2021

@author: jdher
"""
# https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python
import os.path
import numpy as np
import gensim
from gensim import corpora
from gensim.models import LsiModel, Word2Vec
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

def preprocess_data(doc_set):
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
    texts = []
    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        # add tokens to list
        texts.append(stemmed_tokens)
    return texts

def prepare_corpus(doc_clean):
    """
    Input  : clean document
    Purpose: create term dictionary of our courpus and Converting list of documents (corpus) into Document Term Matrix
    Output : term dictionary and Document Term Matrix
    """
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    dictionary = corpora.Dictionary(doc_clean)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    # generate LDA model
    return dictionary,doc_term_matrix
def create_gensim_lsa_model(doc_clean,number_of_topics,words):
    """
    Input  : clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
    dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    # generate LSA model
    lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
    print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
    return lsamodel,dictionary,doc_term_matrix

def create_gensim_logentropy_lsa_model(doc_clean,number_of_topics,words):
    """
    https://radimrehurek.com/gensim/models/logentropy_model.html
    Input  : bow clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
    dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    # generate LSA model
    logentropy= gensim.models.logentropy_model.LogEntropyModel(doc_term_matrix, normalize=True)
    logentropy_lsamodel = LsiModel(logentropy[doc_term_matrix], num_topics=number_of_topics, id2word = dictionary)  # train model
    print(logentropy_lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
    return logentropy_lsamodel,logentropy,dictionary,doc_term_matrix

def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop, start=2, step=3):
    """
    Input   : dictionary : Gensim dictionary
              corpus : Gensim corpus
              texts : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LSA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        # generate LSA model
        model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word = dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

documents_list,titles,num_list=load_data("./","finding36279Terms.txt")
texts=preprocess_data(documents_list)

#compute_coherence_values(dictionary,doc_term_matrix,texts,20)
lsa_model,dictionary,doc_term_matrix=create_gensim_lsa_model(texts, 300, 5)#300 from https://radimrehurek.com/gensim/auto_examples/core/run_topics_and_transformations.html#sphx-glr-auto-examples-core-run-topics-and-transformations-py
Word2Vec_model = Word2Vec(texts, vector_size=100, window=5,min_count=1) 
Word2Vec_sims = Word2Vec_model.wv.most_similar(texts[1], topn=10)
#p=Word2Vec_model.wv["pregnancy"]
#w=Word2Vec_model.wv["woman"]
#m=Word2Vec_model.wv["man"]
#print(p-w+m)
#https://radimrehurek.com/gensim/auto_examples/core/run_similarity_queries.html#sphx-glr-auto-examples-core-run-similarity-queries-py
logentropy_lsamodel,logentropy,dictionary,doc_term_matrix=create_gensim_logentropy_lsa_model(texts, number_of_topics=300, words=10)
logentropy_lsa_index = similarities.MatrixSimilarity(logentropy_lsamodel[logentropy[doc_term_matrix]])
lsa_index=similarities.MatrixSimilarity(lsa_model[doc_term_matrix])


# Perhaps use correlation to gauge how different method scores are, is there a linear relationship?
"""cor=0
for i in range(len(documents_list)):
    lsa_sims=lsa_index[lsa_model[doc_term_matrix[i]]]
    logentropy_lsa_sims = logentropy_lsa_index[lsa_model[logentropy[doc_term_matrix[i]]]]  # perform a similarity query against the corpus
    cor+=np.corrcoef(lsa_sims,logentropy_lsa_sims)
cor=1/len(documents_list)*cor
print(cor)"""

# Show zeroth document's top similarity scoring words. (logentropy lsa seems best, as recommended by wikipedia)
def getSims(i,lsa_index=lsa_index,logentropy_lsa_index=logentropy_lsa_index):
    print(documents_list[i])
    lsa_sims=lsa_index[lsa_model[doc_term_matrix[i]]]
    logentropy_lsa_sims = logentropy_lsa_index[logentropy_lsamodel[logentropy[doc_term_matrix[i]]]]  # perform a similarity query against the corpus
    return lsa_sims,logentropy_lsa_sims,i

"""cor=np.corrcoef(lsa_sims,logentropy_lsa_sims)"""

def Top10matches_LSA(lsa_sims):
    print("="*20)
    print("For LSA\n")
    print(" Top 10 matches for %sth document: %s"%(i,documents_list[i]))
    lsa_sims = sorted(enumerate(lsa_sims), key=lambda item: -item[1])
    count=0
    for doc_position, doc_score in lsa_sims:
        if count<=10:
            print(doc_score, documents_list[doc_position])
        count+=1
def Top10matches_LogEntropy_LSA(logentropy_lsa_sims):
    print("="*20)
    print("For LogEntropy LSA\n")
    print(" Top 10 matches for %sth document: %s"%(i,documents_list[i]))
    logentropy_lsa_sims = sorted(enumerate(logentropy_lsa_sims), key=lambda item: -item[1])
    count=0
    for doc_position, doc_score in logentropy_lsa_sims:
        if count<=10:
            print(doc_score, documents_list[doc_position])
        count+=1
lsa_sims,logentropy_lsa_sims,i=getSims(10)
Top10matches_LSA(lsa_sims)
Top10matches_LogEntropy_LSA(logentropy_lsa_sims)

lsa_sims,logentropy_lsa_sims,i=getSims(20)
Top10matches_LSA(lsa_sims)
Top10matches_LogEntropy_LSA(logentropy_lsa_sims)

lsa_sims,logentropy_lsa_sims,i=getSims(1000)
Top10matches_LSA(lsa_sims)
Top10matches_LogEntropy_LSA(logentropy_lsa_sims)

def find_index_from_document(document_original="Convalescence after chemotherapy (finding)"):
    return [i for i in range(len(documents_list)) if documents_list[i]==document_original][0]

lsa_sims,logentropy_lsa_sims,i=getSims(find_index_from_document())
Top10matches_LSA(lsa_sims)
Top10matches_LogEntropy_LSA(logentropy_lsa_sims)

lsa_sims,logentropy_lsa_sims,i=getSims(find_index_from_document("Increased placental secretion of chorionic gonadotropin (finding)"))
Top10matches_LSA(lsa_sims)
Top10matches_LogEntropy_LSA(logentropy_lsa_sims)
