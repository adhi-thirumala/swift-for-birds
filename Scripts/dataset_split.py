import os
import random
import shutil



"""variables"""

#makesure is a number btw 1 and 100
trainPercent = 80
originalDir = "/media/adhi/Warzone & other Big Games/Swift Project Images/madness"
targetDir = "/media/adhi/Warzone & other Big Games/Swift Project Images/ting/dove2"


def seperate(og, target):
	listOfFiles = os.listdir(og)
	length = len(listOfFiles)
	current = 0
	path = os.path.join(target, "train")
	path2 = os.path.join(target, "val")
	os.mkdir(path)
	os.mkdir(path2)
	while current < length:
		if current <= (trainPercent / 100) * length:
			shutil.copy(og + "/" + listOfFiles[current], target+"/train/"+str(current)+".jpg")
			print(listOfFiles[current])
		else:
			shutil.copy(og + "/" + listOfFiles[current], target+"/val/"+str(current)+".jpg")
			print(listOfFiles[current])
		current += 1


seperate(originalDir, targetDir)
