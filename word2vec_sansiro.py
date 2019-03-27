# For Japanese decoding and encoding
import codecs
import re

fOpen = open('C:/testData/word2vec/sanshiro.txt', 'r')
fOpenContent = fOpen.read()
fOpen.close()

print("Test starts!")

fOpenContent = re.sub(r'[[|「」&!()（）[]$@#":、,：…-『』＃.+?《.+?》]', "", fOpenContent)

#delete special tokens
text = fOpenContent.replace(r'[[|「」&!()（）[]$@#":、,：…-『』＃.+?《.+?》]', "")
print(fOpenContent)
print("The following is text.")
#print(text)

    
