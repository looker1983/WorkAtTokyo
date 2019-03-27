coding="utf-8"
import MeCab
import sys
import datetime
import codecs
import re
import pandas as pd
import numpy as np


#The following method can delete all of the special tokens quoted.
def DeleteSpecialTokens(myLine):
    myLine = myLine.replace(r'[「」&!()（）[]$@#":、,：…-『』]',"")
    print(myLine)

    return myLine
    pass

#The following method can split the sentence without special tokens.
def testMecab(filePath):
    ISOTIMEFORMAT = '%Y-%m%d-%H%M'
    myTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)+".txt"
    myFile = codecs.open(filePath,"r", encoding="utf-8")
    myNewFile = codecs.open(filePath.rstrip(".txt") + myTime,"w",encoding="utf-8",errors="strict")

    myFile.seek(0,0)
    myNewFile.seek(0,0)
    #分词，而且带词性标注。
    #mecab = MeCab.Tagger ("-Ochasen")
    #分词，但是不带词性标注。
    mecab = MeCab.Tagger ("-Owakati")
    

    line = myFile.readline()
    line = DeleteSpecialTokens(line)
    
    print("Test starts!")

    while line != "":
        lineSplit = line.split('\t')
        #print("The title is: " + lineSplit[0])
        sentenceSplit = str(lineSplit[1]).split(r'。')
        for sentence in sentenceSplit:
            #print(sentence)
            mySentence = mecab.parse(sentence)
            myNewFile.write(mySentence)
        
        line = myFile.readline()
        line = DeleteSpecialTokens(line)
    
    print("Test ends!")

    pass    

#myFilePath_ = "/data/txtcombinedWIKI.txt"
#myFilePath_ = "C:/Users/ke_lu/Documents/MyPythonLib/wikiextractor/extracted/AA/final-test/combinedWIKI.txt"
myFilePath_ = "C:/Users/ke_lu/Documents/MyPythonLib/wikiextractor/extracted/AA/final-test/test-temp.txt"
#myFilePath_ = "C:/testMecab.txt"
#myFilePath_ = "C:/test.txt"
testMecab(myFilePath_)




'''
            myData = pd.DataFrame(np.arange(24).reshape((6,4)),index=[1,2,3,4,5,6], columns=['A','B','C','D'])
            print(myData)
'''

