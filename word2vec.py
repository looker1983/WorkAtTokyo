import MeCab
import sys
import datetime
import codecs
import re

#mecab = MeCab.Tagger ("-Ochasen")

#print(mecab.parse("MeCabを用いて文章を分割してみます。"))

#The follwing method will delete all of the empty lines.
def word2vecPre(filePath):
    ISOTIMEFORMAT = '%Y-%m%d-%H%M'
    myTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)+".txt"
    myFile = codecs.open(filePath,"r", encoding="utf-8")
    myNewFile = codecs.open(filePath.rstrip(".txt") + myTime,"w",encoding="utf-8",errors="strict")

    myFile.seek(0,0)
    myNewFile.seek(0,0)

    for line in myFile.readlines():
        lineContent = line.strip('\n')
        if len(lineContent) != 0:
            if lineContent.find("</doc>") != -1:
                myNewFile.write('\n')
                print(lineContent)
                lineContent = myFile.readline().strip() 
        
            if lineContent.find("<doc id=") != -1:
                lineContent = myFile.readline().strip('\n')
                myNewFile.write(line)
                myNewFile.write('\t')
                print(lineContent)
                lineContent = myFile.readline().strip('\n')
            else:
                myNewFile.write(line)
                line = myFile.readline().strip('\n')
        
            
    myFile.close()
    myNewFile.close()

#The file path: "C:\test\testWord2Vec.txt"
#myFilePath_ = "C:/Users/ke_lu/Documents/MyPythonLib/wikiextractor/extracted/AA/wiki_02"
myFilePath_ = "C:/test.txt"
word2vecPre(myFilePath_)


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