# coding: utf-8
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


def word_and_class(doc):
    tagger = MeCab.Tagger('-Ochasen -d /usr/lib/mecab/dic/mecab-ipadic-neologd')
    tagger.parse("")
    doc=re.sub(r'[\d\:\/\#]','',doc)
    result=tagger.parseToNode(doc)
    word_class = []
    while result:
        word = result.surface#.decode("utf-8", "ignore")
        if word.encode('utf-8').isalpha():
            result = result.next
            continue
        clazz = result.feature.split(',')[0]#.decode('utf-8', 'ignore') 
        if (len(result.feature.split(','))>7) and (clazz != u'BOS/EOS') and (clazz in ['名詞', '形容詞', '動詞', '副詞', '助詞',"連体詞", "感動詞"]):
            word_class.append(word)
        result = result.next
    return word_class

    

def process(source_filename, dest_countmat_filename, dest_vocabulary_filename):
    with codecs.open(source_filename,'rb', encoding='utf-8')\
    as f:
        analysis_use = f.readlines()
        print(f)
        print("The type of f is {}.".format(type(f)))
        print(analysis_use)
        print("The typd of analysis_use is {}".format(type(analysis_use)))
        for line in analysis_use:
            print(line)
            print("The type of line is {}".format(type(line)))
            pass

    MAX_FEATURE = None

    #tf=TfidfVectorizer(max_df=1.0, min_df=5, analyzer=word_and_class, use_idf=False, max_features=MAX_FEATURE)
    #cn=CountVectorizer(max_df=1.0, min_df=5, analyzer=lambda doc:doc, max_features=MAX_FEATURE)
    cn=CountVectorizer()
    #print(type(cn))
    #print(cn)
    #tf_matrix=tf.fit_transform(analysis_use['text'])
    count_mat=cn.fit_transform(analysis_use)
    print(cn.get_feature_names())
    print(count_mat.toarray())
    print(cn.vocabulary_)

    with open(dest_countmat_filename, 'wb') as f:
        pickle.dump(count_mat,f,protocol=4)
    with open(dest_vocabulary_filename, 'wb') as d:
        pickle.dump(cn.vocabulary_, d)
    

my_source_filename = "C:/Users/ke_lu/Documents/MyPythonLib/wikiextractor/extracted/AA/final-test/test_count.txt"
my_dest_countmat_filename = "C:/Users/ke_lu/Documents/MyPythonLib/wikiextractor/extracted/AA/final-test/test_count_dest_countmat.txt"
my_dest_vocabulary_filename = "C:/Users/ke_lu/Documents/MyPythonLib/wikiextractor/extracted/AA/final-test/test_count_dest_vocabulary_filename.txt"

#makeList(my_source_filename)
process(my_source_filename,my_dest_countmat_filename,my_dest_vocabulary_filename)


#if __name__ == "__main__":
#    process(sys.argv[1],sys.argv[2],sys.argv[3])
