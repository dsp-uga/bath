
# coding: utf-8

# In[2]:


import scipy as sp
import scipy.misc, scipy.ndimage.interpolation
from medpy import metric
import numpy as np
import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Input,merge, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D,Dropout,Conv2DTranspose
from keras.layers.normalization import BatchNormalization as bn
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from keras import regularizers 
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np 
import nibabel as nib
CUDA_VISIBLE_DEVICES = [0]
os.environ['CUDA_VISIBLE_DEVICES']=','.join([str(x) for x in CUDA_VISIBLE_DEVICES])
#oasis files 1-457

path='/home/bahaa/oasis_mri/OAS1_'


# In[3]:



#Dice coeff
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# In[7]:


#define the model

#define the model
def UNet(input_shape,learn_rate=1e-3):
    l2_lambda = 0.0002
    DropP = 0.3
    kernel_size=3

    inputs = Input(input_shape)

    conv1 = Conv2D( 32, (kernel_size, kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(inputs)
    
    
    conv1 = bn()(conv1)
    
    conv1 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv1)

    conv1 = bn()(conv1)
    
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    pool1 = Dropout(DropP)(pool1)





    conv2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool1)
    
    conv2 = bn()(conv2)

    conv2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv2)

    conv2 = bn()(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    pool2 = Dropout(DropP)(pool2)



    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool2)

    conv3 = bn()(conv3)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv3)
    
    conv3 = bn()(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    pool3 = Dropout(DropP)(pool3)



    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool3)
    conv4 = bn()(conv4)
    
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv4)
    
    conv4 = bn()(conv4)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    pool4 = Dropout(DropP)(pool4)



    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool4)
    
    conv5 = bn()(conv5)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv5)

    conv5 = bn()(conv5)
    
    up6 = concatenate([Conv2DTranspose(256,(2, 2), strides=(2, 2), padding='same')(conv5), conv4],name='up6', axis=3)

    up6 = Dropout(DropP)(up6)


    conv6 = Conv2D(256,(3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up6)
    
    conv6 = bn()(conv6)

    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv6)

    conv6 = bn()(conv6)

    up7 = concatenate([Conv2DTranspose(128,(2, 2), strides=(2, 2), padding='same')(conv6), conv3],name='up7', axis=3)

    up7 = Dropout(DropP)(up7)

    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up7)

    conv7 = bn()(conv7)
    
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv7)

    conv7 = bn()(conv7)

    up8 = concatenate([Conv2DTranspose(64,(2, 2), strides=(2, 2), padding='same')(conv7), conv2],name='up8', axis=3)

    up8 = Dropout(DropP)(up8)

    conv8 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up8)

    conv8 = bn()(conv8)

    
    conv8 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv8)

    conv8 = bn()(conv8)

    up9 = concatenate([Conv2DTranspose(32,(2, 2), strides=(2, 2), padding='same')(conv8), conv1],name='up9',axis=3)

    up9 = Dropout(DropP)(up9)

    conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up9)
    
    conv9 = bn()(conv9)

    conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv9)
   
    conv9 = bn()(conv9)
   
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    return model
 


# In[8]:


model=UNet(input_shape=(512,512,1))
print(model.summary())


# In[62]:




X_train=np.load("X_train.npy")
X_train=X_train.reshape(X_train.shape+(1,))
y_train=np.load("y_train.npy").reshape(X_train.shape)
print('done')




# In[10]:





#training network
model.fit([X_train], [y_train],
                    batch_size=2,
                    nb_epoch=30,
                    #validation_data=([X2_validate],[y_validate]),
                    shuffle=True),
                    #callbacks=[xyz],
                    #class_weight=class_weightt)


# In[29]:



import h5py
#model.save_weights("basic_unet_weights.h5")
model.save('basic_unet_dsp_p3.h5')


# In[30]:
'''


predicted=model.predict(X_test,batch_size=8)
print('done')


# In[35]:


print(predicted.shape)


# In[36]:


import cv2
for i in range(0,len(X_test)):
    cv2.imwrite("large_section_output/"+str(i)+".png",y_test[i])
    cv2.imwrite("large_section_predict/"+str(i)+".png",predicted[i])
print ("done")
    


# In[49]:


smooth=1
def dice_calc(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)


# In[50]:


y_test=np.array(y_test)
predicted=np.array(predicted)


# In[51]:


dice=0
for i in range(0,len(X_test)):
    dice=dice+dice_calc(y_test[i],predicted[i])
print(dice)
    


# In[48]:


print (dice/len(X_test))
'''