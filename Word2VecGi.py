
from nltk.corpus import stopwords 
import pickle
import re


print("Test starts!")


fileName = 'C:/testData/word2vec/train_v2.tsv'

with open(fileName) as myFileOpen:
    myText = myFileOpen.readline()
    while myText != "":
        myText = re.sub(r'[[|「」&!*()（）[]$@#":、,：…-『』＃.+?《.+?》]', "", myText)
        myText = myText.split()
        print("The original text is " + str(myText) + ".")
        words = stopwords.words('english')
        print(words)
        stopWordDic = set(stopwords.words('english'))
        myText = [word for word in myText if word not in stopWordDic]
        print("The text without stop words is " + str(myText) + ".")
        myText = myFileOpen.readline()
print("Test ends!")
    
















def loadFile(obj,fileName):
    myFile = open(fileName,'rb')
    pickle.load(obj,fileName)
    myFile.close()

def saveFile(obj,fileName):
    myFile = open(fileName,'wb')
    pickle.dump(obj,myFile)
    myFile.close()
    

