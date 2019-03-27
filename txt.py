import string, re

title_txt = open('c:/1D-CNN-result for depression.txt&,' 'r+')
try:
    lines = title_txt.readlines()
    for l in lines:
        if l.find("Test Accuracy of the model") != -1:
            l.strip()
except:
    print("There is something wrong.")
finally:
    title_txt.close()

print