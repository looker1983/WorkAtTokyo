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

def word_and_class(doc):
    tagger = MeCab.Tagger('-Ochasen -d /usr/lib/mecab/dic/mecab-ipadic-neologd')
    tagger.parse("")
    doc=re.sub(r'[「」&!()（）[]$@#":、,：…-『』]','',doc)
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
    with open(source_filename,'rb') as f:
        analysis_use = pickle.load(f)

    MAX_FEATURE = 10000#None

    tf=TfidfVectorizer(max_df=1.0, min_df=5, analyzer=word_and_class, use_idf=False, max_features=MAX_FEATURE)
    cn=CountVectorizer(max_df=1.0, min_df=5, analyzer=word_and_class, max_features=MAX_FEATURE)
    tf_matrix=tf.fit_transform(analysis_use['text'])
    count_mat=cn.fit_transform(analysis_use['text'])

    with open(dest_countmat_filename, 'wb') as f:
        pickle.dump(count_mat,f)
    with open(dest_vocabulary_filename, 'wb') as d:
        pickle.dump(cn.vocabulary_, d)


if __name__ == "__main__":
    process(sys.argv[1],sys.argv[2],sys.argv[3])
