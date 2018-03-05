import numpy as np
from PIL import Image
import glob
from operator import add
import cv2

path="/home/rdey/DSP_p3/DSP_003/"
#path="D:\DSP_003/"

X_train=[]
for i in range(0,19):
	print(i)
	counter=0
	temp=[]
	var_to_add=np.zeros((512,512), dtype=np.float32)
	for file in sorted(glob.glob(path+str(i)+"/images/*.tiff")):
		#print(file)
		im = ((np.array(Image.open(file))).astype(np.float32))
		im=im/np.amax(im)
		im=cv2.resize(im,(512, 512))
		for j in range(0,512):
			for k in range(0,512):
				if(im[j,k]>0):

					var_to_add[j,k]=(((var_to_add[j,k])+(im[j,k]))/2.0)
				
		

	X_train.append(var_to_add)
	print(np.array(X_train).shape)
	
	cv2.imwrite(str(i)+".png",(var_to_add/np.amax(var_to_add))*255)

np.save("X_train1",np.array(X_train))



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
np.save("y_train",np.array(y_test))

path="/home/rdey/DSP_p3/DSP_003/test/"
X_train=[]
for i in range(1,10):
	print(i)
	counter=0
	temp=[]
	var_to_add=np.zeros((512,512), dtype=np.float32)
	for file in sorted(glob.glob(path+str(i)+"/images/*.tiff")):
		#print(file)
		im = ((np.array(Image.open(file))).astype(np.float32))
		im=im/np.amax(im)
		im=cv2.resize(im,(512, 512))
		for j in range(0,512):
			for k in range(0,512):
				if(im[j,k]>0):

					var_to_add[j,k]=(((var_to_add[j,k])+(im[j,k]))/2.0)
				
		
		

	X_train.append(var_to_add)
	print(np.array(X_train).shape)
	
	cv2.imwrite(str(i)+".png",(var_to_add/np.amax(var_to_add))*255)

np.save("X_test1",np.array(X_train))








