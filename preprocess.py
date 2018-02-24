import numpy as np
from PIL import Image
import glob
from operator import add
path="D:/DSP_003/"
X_train=[]
for i in range(0,19):
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





