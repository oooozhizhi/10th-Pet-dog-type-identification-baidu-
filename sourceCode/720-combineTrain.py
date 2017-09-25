import h5py
import numpy as np
import keras
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *
from keras.utils.np_utils import to_categorical
import pandas as pd
from keras.preprocessing.image import *
from keras import backend as K
'''
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu':0})))
'''

np.random.seed(2017)

X_train = []
X_test = []
'''
"/your path/0714_512-512_deng_gap_InceptionV3.h5",\
"/your path/0714_512-512_deng_gap_Xception.h5",\

"/your path/0715_512-512_crop_gap_InceptionV3.h5",\
"/your path/0715_512-512_crop_gap_Xception.h5",\

"/your path/0718_512-512_crop(zo_sh)_gap_Xception.h5",\
"/your path/0718_512-512_crop(zo_sh)_gap_InceptionV3.h5"

"/your path/0720_512_(non-Random_batch8)_train_InceptionV3.h5",\
				"/your path/0720_512_(non-Random_batch8)_train_Xception.h5",\
'''
gen = ImageDataGenerator()
train_generator = gen.flow_from_directory("your path", (512,512), shuffle=False, batch_size=8)
y_train=train_generator.classes

for filename in ["/your path/448-res2.h5"]:
	with h5py.File(filename, 'r') as h:
		X_train.append(np.array(h['train']))
		X_test.append(np.array(h['test']))
		y_train = np.array(h['label'])

for filename in [
				"/your path/224torch1p_resnet101",
				"/your path/448-res2.h5",
				"/your path/224torch1p_densenet161",
				"/your path/224torch1p_densenet201",				
				"/your path/299-incepV4.h5",
				"/your path/512-Xception.h5"]:
	with h5py.File(filename, 'r') as h:
		X_train.append(np.array(h['train']))
		X_test.append(np.array(h['test']))
		#y_train = np.array(h['label'])
'''
for filename in ["/your path/0719_1024_non-crop(batch8)_test_InceptionV3.h5",\
"/your path/0719_1024_non-crop(batch8)_test_Xception.h5"]:
	with h5py.File(filename, 'r') as h:
		#X_train.append(np.array(h['train']))
		X_test.append(np.array(h['test']))
		#y_train = np.array(h['label'])
X_test = np.concatenate(X_test, axis=1)
'''
X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)
y_train = to_categorical(y_train, num_classes=100)
X_train, y_train = shuffle(X_train, y_train)


input_tensor = Input(X_train.shape[1:])
x = Dense(512,activation='relu')(input_tensor)
x = Dropout(0.5)(x)
x = Dense(100,activation='softmax')(x)
model = Model(input_tensor,x)



model.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=64,nb_epoch=12, validation_split=0.0,verbose=True)

keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0001,nesterov=True)
model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32,nb_epoch=20, validation_split=0.2,verbose=True)

keras.optimizers.SGD(lr=0.001, momentum=0.9,decay=0.0001,nesterov=True)
model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128,nb_epoch=10, validation_split=0.2,verbose=True)

keras.optimizers.SGD(lr=0.0001, momentum=0.9,decay=0.0001,nesterov=True)
model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128,nb_epoch=20, validation_split=0.2,verbose=True)


#keras.optimizers.SGD(lr=3.0,momentum=0.0,decay=0.005,nesteroy=False)
#model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
#model.fit(X_train,y_train,batch_size=256,nb_epoch=1000,validation_split=0.2,verbose=True)


y_pred = model.predict(X_test, verbose=1)

y_class=[]
for i in range(10593):
    y_class.append(y_pred[i].argmax())
K.clear_session()


gen = ImageDataGenerator()
test_generator = gen.flow_from_directory("/your path/test", (512,512), shuffle=False, batch_size=8, class_mode=None)

train_generator = gen.flow_from_directory("/your path/train", (512,512), shuffle=False, batch_size=8)
a=train_generator.class_indices
map={}
for i in a:
    map[a[i]]=i.split('_')[1]

re=[]
for i, fname in enumerate(test_generator.filenames):
    fname = fname.split('.')[0].split('/')[1]
    re.append([fname,map[y_class[i]]])

df=pd.DataFrame(re)

df.to_csv('/your path/nocut_res152.csv', index=None)

