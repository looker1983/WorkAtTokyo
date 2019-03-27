import datetime


print("Wave test starts!\n")
startTime = datetime.datetime.now()

targetFile = 'AB001-1_Pt_free talk_20171026_110943_071.wav'
targetFolder = 'F:/Riku_Ka/testdata/AB001-1_voice/'
resultFolder = 'F:/Riku_Ka/testdata/AB001-1_voice/result/'




endTime = datetime.datetime.now()
print("\nTest ends! The time spent: " + str(endTime - startTime))
