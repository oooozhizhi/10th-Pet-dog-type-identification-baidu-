import h5py
import numpy as np
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *
from keras.utils.np_utils import to_categorical
import pandas as pd
from keras.preprocessing.image import *
np.random.seed(2017)

X_train = []
X_test = []
for filename in [ "/your path/gap_Xception.h5"]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        y_train = np.array(h['label'])

X_train = np.concatenate(X_train, axis=1)
y_train = to_categorical(y_train, num_classes=100)
#X_train, y_train = shuffle(X_train, y_train)
print(X_train.shape[1:])
'''
input_tensor = Input(X_train.shape[1:])
x = Dense(256,activation='relu')(input_tensor)
x = Dropout(0.5)(x)
x = Dense(100, activation='softmax')(x)
model = Model(input_tensor, x)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=16, nb_epoch=30, validation_split=0.2,verbose=True)
y_pred = model.predict(X_test, verbose=1)

y_class=[]
for i in range(10593):
    y_class.append(y_pred[i].argmax())



gen = ImageDataGenerator()
test_generator = gen.flow_from_directory("/your path/test", (224, 224), shuffle=False, batch_size=16, class_mode=None)

train_generator = gen.flow_from_directory("/your path/traindata", (224, 224), shuffle=False, batch_size=16)
a=train_generator.class_indices
map={}
for i in a:
    map[a[i]]=i.split('_')[1]

re=[]
for i, fname in enumerate(test_generator.filenames):
    fname = fname.split('.')[0].split('/')[1]
    re.append([fname,map[y_class[i]]])

df=pd.DataFrame(re)

df.to_csv('/your path/pred.csv', index=None)
'''