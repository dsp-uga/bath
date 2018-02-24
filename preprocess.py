import numpy as np
from PIL import Image
import glob
from operator import add

path="/home/rdey/DSP_p3/DSP_003/"
#path="D:\DSP_003/"

X_train=[]
for i in range(0,19):
	print(i)
	counter=0
	temp=[]
	var_to_add=np.zeros((512,512), dtype=np.float32).tolist()
	for file in glob.glob(path+str(i)+"/images/*.tiff"):
		im = ((np.array(Image.open(file))).astype(np.float32)).tolist()
		var_to_add=map(add,var_to_add,im)
		if(counter==250 or counter==(len(glob.glob(path+str(i)+"/images/*.tiff"))-1)):
			temp.append(var_to_add)
			var_to_add=np.zeros((512,512), dtype=np.float32).tolist()

	X_train.append(temp)
np.save("X_train",np.array(X_train))



###############################################################################################
import json
import glob
import numpy as np
import cv2
y_test=[]
for i in range(0,19):
	print(i)

	mask=np.zeros((512,512), dtype=np.float32)
	data = json.load(open(path+str(i)+"/regions/regions.json"))
	for j in range(0,len(data)):
		#print(data[j]['coordinates'])
		for k in range(0,len(data[j]['coordinates'])):

			mask[(data[j]['coordinates'][k][0]),(data[j]['coordinates'][k][1])]=1
	y_test.append(mask)
	cv2.imwrite("mask"+str(i)+".png",mask*255)







