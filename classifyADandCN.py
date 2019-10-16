import numpy as np
import scipy.io as scird
import tensorflow as tf
from tensorflow import keras

#read the data
cnmat = scird.loadmat('CNcounts141.mat')
CNcounts=cnmat['CNcounts']
admat = scird.loadmat('ADcounts141.mat')
ADcounts=admat['ADcounts']

#print arrays sizes
print(ADcounts.shape)
print(CNcounts.shape)
#both: (141, 256, 160) which means:
#we have 141 images per each class 
#each image has 160 slices
#256 is number of hisrogram counts

#now consider slice 70
s70ad= ADcounts[:,:,70]
s70cn= CNcounts[:,:,70]

#normalize arrays
s70ad= s70ad.astype(float)
for rw in range(0,141):
    s70ad[rw,:]=s70ad[rw,:]/ max(s70ad[rw,:])
s70cn= s70cn.astype(float)
for rw in range(0,141):
    s70cn[rw,:]=s70cn[rw,:]/ max(s70cn[rw,:])
#double check sizes
print(s70cn.shape)
# result is (141, 256) which means 141 images and 256 hist-counts

#now we build training matrix; 184: 92 AD 92 CN
xtrn = np.zeros([184,256])
xtrn[0:92,:] = s70ad[0:92,:]
xtrn[92:,:] = s70cn[0:92,:]

#next the labels
ytrn = [0]*184
ytrn[92:]=[1]*92
ytrn = np.asarray(ytrn)

#so now we are ready for building te model
#next few lines ignored because we don't need to flatten the data
'''
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
'''

#that's our model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#fit the model to our trainnig data
model.fit(xtrn,ytrn, epochs=10)

#prepare test data; 98: 49 AD 49 CN
xtst= np.zeros([98,256])
xtst[0:49,:] = s70ad[92:,:]
xtst[49:,:] = s70cn[92:,:]

#test data labels
ytst= [0]*98
ytst[49:]=[1]*49

#finally evaluate the model and make predictions
test_loss, test_acc = model.evaluate(xtst, ytst, verbose=2)
print(test_acc)

predictions = model.predict(xtst)
print(predictions[0])
print(predictions[1])
print(predictions[2])
print(predictions[3])
print(predictions[4])