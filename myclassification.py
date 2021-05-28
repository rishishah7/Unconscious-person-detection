#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from random import shuffle
import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as K
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
print(tf.__version__)
print(keras.__version__)
import keras_utils

def reset_tf_session():
    curr_session = tf.compat.v1.get_default_session()
    # close current session
    if curr_session is not None:
        curr_session.close()
    # reset graph
    K.clear_session()
    # create new session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    s = tf.compat.v1.InteractiveSession(config=config)
    #K.set_session(s)
    tf.compat.v1.keras.backend.set_session(s)
    return s


# In[2]:


data = 'C:/Users/Harsh/intro-to-dl/week3/data'


# In[3]:


def ohev(img):
    label = img.split('.')[-3]
    if label == 'c': return [0]
    elif label == 'uc':return [1]
    


# In[4]:


data2 = []
def creatdata():

    for aa in tqdm(os.listdir(data)):
        label = ohev(aa)
        path = os.path.join(data,aa)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        #img = cv2.resize(img, (30,30))
        data2.append([img,label])
    shuffle(data2)
    np.save('data.npy',data2)
    
        
                     
    return data2


# In[5]:


olol = creatdata();


# In[6]:


train = olol[:-20]
crossv = olol[-20:]



Xtrain = np.array([i[0] for i in train])
Ytrain = [i[1] for i in train]

Xcrossv = np.array([i[0] for i in crossv])
Ycrossv = [i[1] for i in crossv]

x_train2 = (Xtrain/255) - 0.5
x_crossv2 = (Xcrossv/255) - 0.5

y_train2 = keras.utils.to_categorical(Ytrain,2)
y_crossv2 = keras.utils.to_categorical(Ycrossv,2)
print(y_train2)


# In[ ]:





# In[7]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout ,InputLayer
from keras.layers.advanced_activations import LeakyReLU


# In[8]:


def make_model():
    
    model = Sequential()
    model.add(InputLayer([300,300,3]))
    
    model.add(Conv2D(16,[3,3],padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(32,[3,3],padding = 'same'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D([2,2]))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(32,[3,3],padding = 'same'))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(64,[3,3],padding = 'same'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D([2,2]))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(5,))
    model.add(Dropout(0.5))
    model.add(LeakyReLU(0.1))
    model.add(Dense(2,))
    model.add(Activation('softmax'))
    

    
    return model


# In[9]:


s = reset_tf_session()
model = make_model()
model.summary()


# In[10]:


INIT_LR = 5e-3  # initial learning rate
BATCH_SIZE = 2
EPOCHS = 5

s = reset_tf_session()  
model = make_model()  

model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.adamax(lr=INIT_LR),
    metrics=['accuracy']
)


def lr_scheduler(epoch):
    return INIT_LR * 0.9 ** epoch

class LrHistory(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        print("Learning rate:", K.get_value(model.optimizer.lr))


# In[11]:


# we will save model checkpoints to continue training in case of kernel death
model_filename = 'myhack.{0:03d}.hdf5'
last_finished_epoch = None

#### uncomment below to continue training from model checkpoint
#### fill `last_finished_epoch` with your latest finished epoch
# from keras.models import load_model
# s = reset_tf_session()
# last_finished_epoch = 7
# model = load_model(model_filename.format(last_finished_epoch))


# In[12]:


model.fit(
    x_train2, y_train2,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler), 
               LrHistory(), 
               keras_utils.TqdmProgressCallback(),
               keras_utils.ModelSaveCallback(model_filename)],
    validation_data=(x_crossv2, y_crossv2),
    shuffle=True,
    verbose=0,
    initial_epoch=last_finished_epoch or 0
)


# In[13]:


y_pred_test = model.predict_proba(x_crossv2)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)
y_pred_test_max_probas = np.max(y_pred_test, axis=1)


# In[14]:


person = ["consience", "unconscience"]
from sklearn.metrics import confusion_matrix, accuracy_score
plt.figure(figsize=(7, 6))
plt.title('Confusion matrix', fontsize=16)
plt.imshow(confusion_matrix(Ycrossv, y_pred_test_classes))
plt.xticks(np.arange(2), person, rotation=45, fontsize=12)
plt.yticks(np.arange(2), person, fontsize=12)
plt.colorbar()
plt.show()
print("Test accuracy:", accuracy_score(Ycrossv, y_pred_test_classes))


# In[ ]:




