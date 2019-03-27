from pydub import AudioSegment
from pydub.silence import split_on_silence
import datetime


print("Wave test starts!\n")
startTime = datetime.datetime.now()

targetFile = 'AB001-1_Th_free talk_20171026_110943_071.wav'
targetFolder = 'F:/Riku_Ka/testdata/AB001-1_voice/'
resultFolder = 'F:/Riku_Ka/testdata/AB001-1_voice/result/'

originalFile = AudioSegment.from_file(targetFolder + targetFile, format="wav")
chunks = split_on_silence(originalFile, min_silence_len=300, silence_thresh=-55)
print("The length of seperatedVoice is :" + str(len(chunks)))
finalResult = chunks[2:]

for i, chunk in enumerate(chunks):
    chunk.export(resultFolder + targetFile[0:20] + "_" + str(i+1)+ ".wav", format="wav")


endTime = datetime.datetime.now()
print("\nWave test ends! The time spent: " + str(endTime - startTime))
