
# coding: utf-8

# In[2]:

import keras
import scipy as sp
import scipy.misc, scipy.ndimage.interpolation
from medpy import metric
import numpy as np
import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Input,merge, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D,Dropout,Conv2DTranspose,add,multiply
from keras.layers.normalization import BatchNormalization as bn
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from keras import regularizers 
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np 
import nibabel as nib
CUDA_VISIBLE_DEVICES = [1]
os.environ['CUDA_VISIBLE_DEVICES']=','.join([str(x) for x in CUDA_VISIBLE_DEVICES])
#oasis files 1-457

path='/home/bahaa/oasis_mri/OAS1_'


# In[3]:

import numpy as np
import cv2







#Dice coeff
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def neg_dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (1-((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)))


def neg_dice_coef_loss(y_true, y_pred):
    return dice_coef(y_true, y_pred)

  

X_train=np.load("X_train.npy")
X_train=X_train.reshape(X_train.shape+(1,))
y_train=np.load("y_train.npy").reshape(X_train.shape)

from keras.models import load_model
import h5py
#model.save_weights("basic_unet_weights.h5")
#model=load_model('probability_unet_extra_dice_whole_modified.h5', custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef})

model=load_model('basic_unet_dsp_p3_round3.h5', custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef,'neg_dice_coef_loss':neg_dice_coef_loss})

model.fit([X_train], [y_train],
                batch_size=4,
                nb_epoch=1000,
                        #validation_data=([X2_validate],[y_validate]),
                shuffle=True)
                        #callbacks=[xyz],
                        #class_weight=class_weightt)


# In[29]:



import h5py
#model.save_weights("basic_unet_weights.h5")
model.save('basic_unet_dsp_p3_round3.h5')
