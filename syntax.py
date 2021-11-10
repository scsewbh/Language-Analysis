from difflib import SequenceMatcher
import os
from itertools import islice

def load_data(path,file_name):
    documents_list = []
    titles=[]
    num_list=[]
    with open( os.path.join(path, file_name) ,"r") as fin:
        for line in fin.readlines():
            num,text = line.strip().split("\t")
            documents_list.append(text)
            num_list.append(num)
    print("Total Number of Documents:",len(documents_list))
    #print(num_list)
    titles.append( text[0:min(len(text),100)] )
    return documents_list,titles,num_list

documents_list,titles,num_list = load_data("./","finding36279Terms.txt")

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def checkAllSyntax(a):
    values = {} #index+1 is line number
    for line in documents_list:
        values[line] = (similar(a, line))

    N = 10
    res = dict(sorted(values.items(), key=lambda item: item[1], reverse=True))
    for key, value in islice(res.items(), 10):
        print(value, key)

checkAllSyntax("Increased placental secretion of chorionic gonadotropin (finding)")