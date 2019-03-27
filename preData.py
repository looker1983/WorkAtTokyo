# coding: utf-8
import MeCab
import sys
import pandas, numpy, seaborn, pymc3, theano
import pandas, re, matplotlib, numpy
from scipy import sparse
import seaborn, scipy
import sklearn
import sklearn.utils.extmath
import sklearn.model_selection
import pymc3, theano
from pymc3.distributions.transforms import t_stick_breaking
from sklearn.metrics import roc_curve, auc
import MeCab
import sklearn, numpy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Reshape, Activation, Input, Lambda
from keras.preprocessing import sequence
import glob
from keras.preprocessing.text import Tokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import keras
from keras.layers.merge import Dot
from keras.preprocessing.sequence import skipgrams
import tensorflow, gensim
import gensim.downloader
import nltk, pickle
import datetime
import codecs
import re
import pickle



#The follwing method will delete all of the empty lines.
def deleteEmptyLine(original_filename):
    print("Start deleting empty lines!")
    ISOTIMEFORMAT = '-%Y-%m%d-%H%M'
    myTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    
    myFile = codecs.open(original_filename,'r', encoding='utf-8')
    myNewFile = codecs.open(original_filename + '-withOutEmptyLines' + myTime,\
    'w',encoding='utf-8',errors='strict')

    myFile.seek(0,0)
    myNewFile.seek(0,0)

    for line in myFile.readlines():
        line = line.strip()
        if len(line) != 0:
            myNewFile.write(line)
            myNewFile.write("\n")

    myFile.close()
    myNewFile.close()
    print("Empty lines have all been deleted.")
    print("The result file is {}".format(original_filename + '-withOutEmptyLines' + myTime))
    pass


# The following method will preprocess the data.
def preData(original_filename):
    print("Data preprocesing starts!")
    ISOTIMEFORMAT = '-%Y-%m%d-%H%M'
    myTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    myFile = codecs.open(original_filename,'r', encoding='utf-8')
    myFinalFile = codecs.open(original_filename + '-final'+ \
    myTime,'w',encoding='utf-8')

    line = myFile.readline().strip()

    while line != "":
        if line.find("</doc>") != -1:
            myFinalFile.write('\n')
            line = myFile.readline().strip('\n') 
        
        if line.find('<doc id=') != -1:
            line = myFile.readline().strip('\n')
            myFinalFile.write(line)
            myFinalFile.write('\t')
            print(line)
            line = myFile.readline().strip('\n')
        else:
            myFinalFile.write(line)
            print(line)
            line = myFile.readline().strip('\n')

    myFile.close()
    myFinalFile.close()  
    print("Data preprocessing ends!")
    print("The result file is {}".format('-final-' + original_filename + myTime)) 
    pass

#myOriginalFilePath_ = "C:/Users/ke_lu/Documents/MyPythonLib/wikiextractor/extracted/AA/wiki_02"
myOriginalFilePath_ = "C:/Users/ke_lu/Documents/MyPythonLib/wikiextractor/extracted/AA/wiki_02-withOutEmptyLines-2018-1205-1629"


#deleteEmptyLine(myOriginalFilePath_)
preData(myOriginalFilePath_)

'''
if __name__=='__main__':
	if len(sys.argv)==1:
		print("Please input the file path!!!")
else:
    deleteEmptyLine(sys.argv[1])

'''


#import sys
#print(sys.path)


#python WikiExtractor.py -b 1024M -o extracted jawiki-latest-pages-articles.xml.bz2

#mecab = MeCab.Tagger ("-Ochasen")
#print(mecab.parse("MeCabを用いて文章を分割してみます。"))