from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
import h5py
import inceptionV4
import resnet152
import densenet161
import densenet121
import densenet169
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
import numpy as np
import cv2
import os
fileDir = '/your path/'

def write_gap_train(MODEL, image_size, lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)

    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    gen = ImageDataGenerator()
   
    train_generator = gen.flow_from_directory("/your path/train_cut4p", image_size, shuffle=False, batch_size=8)#72897
    train = model.predict_generator(train_generator,9113,verbose=True)#72897
    with h5py.File(fileDir+"your path/512_4p-%s.h5"%MODEL.__name__) as h:
        h.create_dataset("train", data=train)
    '''
    train_generator = gen.flow_from_directory("/your path/train_cut3p_hun", image_size, shuffle=False, batch_size=8)#85675
    train = model.predict_generator(train_generator,10710,verbose=True)#85675
    with h5py.File(fileDir+your path/512_3p-%s.h5"%MODEL.__name__) as h:
        h.create_dataset("train", data=train)
    '''
    #train_generator = gen.flow_from_directory("/your path", image_size, shuffle=False, batch_size=8)#18686
    #train_generator = gen.flow_from_directory("/your path", image_size, shuffle=False, batch_size=8)#37175
    #train_generator = gen.flow_from_directory("/your path", image_size, shuffle=False, batch_size=8)#55228
    
    #train_generator = gen.flow_from_directory("/your path", image_size, shuffle=False, batch_size=8)#18686
    #train_generator = gen.flow_from_directory("/your path", image_size, shuffle=False, batch_size=8)#37178
    ##train_generator = gen.flow_from_directory("/your path", image_size, shuffle=False, batch_size=8)#55298
     
    #test_generator1 = gen.flow_from_directory("/your path", image_size, shuffle=False, batch_size=8, class_mode=None)#29282
    #test_generator2 = gen.flow_from_directory("/your path", image_size, shuffle=False, batch_size=8, class_mode=None)#29282
     
    #train = model.predict_generator(train_generator,2336,verbose=True)#18686
    #train = model.predict_generator(train_generator,4647,verbose=True)#37175
    #train = model.predict_generator(train_generator,6904,verbose=True)#55228
    #train = model.predict_generator(train_generator,3658,verbose=True)#29259
    #train = model.predict_generator(train_generator,6908,verbose=True)#55261
    #val = model.predict_generator(val_generator, 390,verbose=True)#3117
    #train = model.predict_generator(train_generator,4651,verbose=True)#37201
    #train = model.predict_generator(train_generator,6910,verbose=True)#55273
    #train = model.predict_generator(train_generator,9127,verbose=True)#73014
    #train = model.predict_generator(train_generator,9132,verbose=True)#73049
    #train = model.predict_generator(train_generator,9123,verbose=True)#72981
    #train = model.predict_generator(train_generator,11282,verbose=True)#90253
    '''
    test1 = model.predict_generator(test_generator1, 3661,verbose=True)#29282
    with h5py.File(fileDir+"wh_code/test_cut/512-%s.h5"%MODEL.__name__) as h:
        h.create_dataset("test", data=test1)

    test2 = model.predict_generator(test_generator2, 3661,verbose=True)#29282
    with h5py.File(fileDir+"wh_code/test_nocut/512-%s.h5"%MODEL.__name__) as h:
        h.create_dataset("test", data=test2)
    '''
        
    
    

    

    K.clear_session()

def write_train(MODEL, image_size, lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)

    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

  
    gen = ImageDataGenerator()
    batchsize=16
    batch_X = np.zeros((batchsize,)+(width,height,3),dtype=K.floatx())
    train_feature=[]
    gen = ImageDataGenerator()
    train = gen.flow_from_directory("/your path/train", image_size, shuffle=False, batch_size=8)#18686
    postrain = '/your path/'
    for idx in range(0, len(train.filenames),batchsize):
        print(idx)
        if idx + batchsize<len(train.filenames):
            for i in range(batchsize):
                batch_X[i]=get_processed_image(postrain+train.filenames[idx+i].split('/')[1])
            train_feature.append(model.predict_on_batch(batch_X))
        else:
            length = len(train.filenames)-idx
            for i in range(length):
                batch_X[i]=get_processed_image(postrain+train.filenames[idx+i].split('/')[1])
            train_feature.append(model.predict_on_batch(batch_X[0:length]))

    train_feature = np.array(train_feature)
    train_feature = np.concatenate(train_feature, 0)
    
    test_feature=[]
    test = gen.flow_from_directory("/your path/test", image_size, shuffle=False, batch_size=8)#18686
    postest = '/your path/'
    for idx in range(0, len(test.filenames),batchsize):
        print(idx)
        if idx + batchsize<len(test.filenames):
            for i in range(batchsize):
                batch_X[i]=get_processed_image(postest+test.filenames[idx+i].split('/')[1])
            test_feature.append(model.predict_on_batch(batch_X))
        else:
            length = len(test.filenames)-idx
            for i in range(length):
                batch_X[i]=get_processed_image(postest+test.filenames[idx+i].split('/')[1])
            test_feature.append(model.predict_on_batch(batch_X[0:length]))

    test_feature = np.array(test_feature)
    test_feature = np.concatenate(test_feature, 0)

    print(train_feature.shape)
    print(test_feature.shape)
    with h5py.File(fileDir+"your path/299_1p_tou-xcep.h5") as h:
        h.create_dataset("train", data=train_feature)
        h.create_dataset("test", data=test_feature)

def incepV4():
    
    image_size=(299,299)
    base_model = inceptionV4.create_model(weights='imagenet', include_top=False)
    x = AveragePooling2D((8,8), padding='valid')(base_model.output)
    x = Flatten()(x) 
    model = Model(base_model.input,x)
    gen = ImageDataGenerator(preprocessing_function=inceptionV4.preprocess_input)

    #train_generator = gen.flow_from_directory("/your path/train_cut2p", image_size, shuffle=False, batch_size=8)#37189

    #train_generator = gen.flow_from_directory("/your path/traindata2_pre", image_size, shuffle=False, batch_size=8)#20227
    #val_generator = gen.flow_from_directory("/your path", image_size, shuffle=False, batch_size=8)#3117
    #train_generator = gen.flow_from_directory("/your path/train", image_size, shuffle=False, batch_size=8)#18686
    train_generator = gen.flow_from_directory("/your path/train_cut4p", image_size, shuffle=False, batch_size=8)#72897
    train = model.predict_generator(train_generator,9113,verbose=True)#72897
    with h5py.File(fileDir+"your path/299_4p-incepv4.h5") as h:
        h.create_dataset("train", data=train)

    '''
    train_generator = gen.flow_from_directory("/your path/train3p_hun", image_size, shuffle=False, batch_size=8)#85828
    train = model.predict_generator(train_generator,10729,verbose=True)#85828    
    with h5py.File(fileDir+"your path/299_3p-incepv4.h5") as h:
        h.create_dataset("train", data=train)

    train_generator = gen.flow_from_directory("/your path/train_cut3p_hun", image_size, shuffle=False, batch_size=8)#85675
    train = model.predict_generator(train_generator,10710,verbose=True)#85675
    with h5py.File(fileDir+"your path/299_3p-incepV4.h5") as h:
        h.create_dataset("train", data=train)
    '''
    #train_generator = gen.flow_from_directory("/your path/train_cut", image_size, shuffle=False, batch_size=8)#18686
    #train_generator = gen.flow_from_directory("/your path/train",image_size, shuffle=False, batch_size=8)#18686
    #train_generator = gen.flow_from_directory("/your path/train_cut3p", image_size, shuffle=False, batch_size=8)#55298
    #test_generator = gen.flow_from_directory("/your path/test", image_size, shuffle=False, batch_size=8, class_mode=None)#29282
    #test_generator = gen.flow_from_directory("/your path/test_cut", image_size, shuffle=False, batch_size=8, class_mode=None)#10593
    #train_generator = gen.flow_from_directory("/your path/train4p", image_size, shuffle=False, batch_size=8)#73049
    #train_generator = gen.flow_from_directory("/your path/train_cut4p", image_size, shuffle=False, batch_size=8)#72981
    #train_generator = gen.flow_from_directory("/your path/train_cut5p", image_size, shuffle=False, batch_size=8)#90253
    #train_generator = gen.flow_from_directory("/your path/train2p", image_size, shuffle=False, batch_size=8)#37175
    #train_generator = gen.flow_from_directory("/your path/train3p", image_size, shuffle=False, batch_size=8)#55228
    #train_generator = gen.flow_from_directory("/your path/train_hun", image_size, shuffle=False, batch_size=8)#29259
    #train = model.predict_generator(train_generator,2336,verbose=True)#18686
    
    #test_generator2 = gen.flow_from_directory("/your path/test", image_size, shuffle=False, batch_size=8, class_mode=None)#29282
    #train = model.predict_generator(train_generator,4647,verbose=True)#37175
    #train = model.predict_generator(train_generator,6904,verbose=True)#55228
    #train = model.predict_generator(train_generator,3658,verbose=True)#29259
    #train = model.predict_generator(train_generator,6908,verbose=True)#55261
    #val = model.predict_generator(val_generator, 390,verbose=True)#3117
    #train = model.predict_generator(train_generator,4651,verbose=True)#37201
    #train = model.predict_generator(train_generator,6910,verbose=True)#55273
    #train = model.predict_generator(train_generator,9127,verbose=True)#73014
    #train = model.predict_generator(train_generator,9123,verbose=True)#72981
    #train = model.predict_generator(train_generator,11282,verbose=True)#90253
    #test = model.predict_generator(test_generator, 3661,verbose=True)#29282
    
    #print(test .shape)
    
        #h.create_dataset("test", data=test)
        #h.create_dataset("label", data=train_generator.classes)
        #h.create_dataset("val", data=val)
        #h.create_dataset("trainlabel", data=train_generator.classes)
        #h.create_dataset("vallabel", data=val_generator.classes)
   
    K.clear_session()



def get_processed_image(img_path):
    # Load image and convert from BGR to RGB
    im = np.asarray(cv2.imread(img_path))[:,:,::-1]
    im = cv2.resize(im, (512, 512))
    #im = inceptionV4.preprocess_input(im)
    return im

def incepV4_2():
    base_model = inceptionV4.create_model(weights='imagenet', include_top=False)
    x = AveragePooling2D((8,8), padding='valid')(base_model.output)
    x = Flatten()(x) 
    model = Model(base_model.input,x)

    batchsize=16
    image_size=(299,299)
    batch_X = np.zeros((batchsize,)+(299,299,3),dtype=K.floatx())
    train_feature=[]
    gen = ImageDataGenerator()
    train = gen.flow_from_directory("/your path/train_hun", image_size, shuffle=False, batch_size=8)#18686
    postrain = '/your path/train_hun_head/'
    for idx in range(0, len(train.filenames),batchsize):
        print(idx)
        if idx + batchsize<len(train.filenames):
            for i in range(batchsize):
                batch_X[i]=get_processed_image(postrain+train.filenames[idx+i].split('/')[1])
            train_feature.append(model.predict_on_batch(batch_X))
        else:
            length = len(train.filenames)-idx
            for i in range(length):
                batch_X[i]=get_processed_image(postrain+train.filenames[idx+i].split('/')[1])
            train_feature.append(model.predict_on_batch(batch_X[0:length]))

    train_feature = np.array(train_feature)
    train_feature = np.concatenate(train_feature, 0)
    
    test_feature=[]
    test = gen.flow_from_directory("/your path/test", image_size, shuffle=False, batch_size=8)#18686
    postest = '/your path/test_head/'
    for idx in range(0, len(test.filenames),batchsize):
        print(idx)
        if idx + batchsize<len(test.filenames):
            for i in range(batchsize):
                batch_X[i]=get_processed_image(postest+test.filenames[idx+i].split('/')[1])
            test_feature.append(model.predict_on_batch(batch_X))
        else:
            length = len(test.filenames)-idx
            for i in range(length):
                batch_X[i]=get_processed_image(postest+test.filenames[idx+i].split('/')[1])
            test_feature.append(model.predict_on_batch(batch_X[0:length]))

    test_feature = np.array(test_feature)
    test_feature = np.concatenate(test_feature, 0)

    print(train_feature.shape)
    print(test_feature.shape)
    with h5py.File('/your path/wh_code/train_head_hun/299-incepV4.h5', "w") as f:
         f.create_dataset("train", data=train_feature)
    #f.create_dataset("test", data=test_feature)
    #f.create_dataset("label", data=train.classes)
    with h5py.File('/your path/wh_code/test_head/299-incepV4.h5', "w") as f:
         f.create_dataset("test", data=test_feature)
    #train_generator = gen.flow_from_directory("/your path/train_cut2p", image_size, shuffle=False, batch_size=8)#37189

    #train_generator = gen.flow_from_directory("/your path/traindata2_pre", image_size, shuffle=False, batch_size=8)#20227
    #val_generator = gen.flow_from_directory("/your path/val_cutpre", image_size, shuffle=False, batch_size=8)#3117
    #train_generator = gen.flow_from_directory("/your path/train", image_size, shuffle=False, batch_size=8)#18686
    
    '''
    #train_generator = gen.flow_from_directory("/your path/train_cut2p", image_size, shuffle=False, batch_size=8)#37189
    #train_generator = gen.flow_from_directory("/your path/train_cut", image_size, shuffle=False, batch_size=8)#18686
    #train_generator = gen.flow_from_directory("/your path/train",image_size, shuffle=False, batch_size=8)#18686
    #train_generator = gen.flow_from_directory("/your path/train_cut3p", image_size, shuffle=False, batch_size=8)#55298
    #test_generator = gen.flow_from_directory("/your path/test", image_size, shuffle=False, batch_size=8, class_mode=None)#10593
    test_generator = gen.flow_from_directory("/your path/test_cut", image_size, shuffle=False, batch_size=8, class_mode=None)#10593
    #train_generator = gen.flow_from_directory("/your path/train4p", image_size, shuffle=False, batch_size=8)#73049
    #train_generator = gen.flow_from_directory("/your path/train_cut4p", image_size, shuffle=False, batch_size=8)#72981
    train_generator = gen.flow_from_directory("/your path/train_cut5p", image_size, shuffle=False, batch_size=8)#90253
    #train_generator = gen.flow_from_directory("/your path/train2p", image_size, shuffle=False, batch_size=8)#37201
    #train_generator = gen.flow_from_directory("/your path/train3p", image_size, shuffle=False, batch_size=8)#55273
    #train = model.predict_generator(train_generator,2336,verbose=True)#18686
    #train = model.predict_generator(train_generator,4649,verbose=True)#37189
    #train = model.predict_generator(train_generator,6913,verbose=True)#55298
    #train = model.predict_generator(train_generator,6908,verbose=True)#55261
    #val = model.predict_generator(val_generator, 390,verbose=True)#3117
    #train = model.predict_generator(train_generator,4651,verbose=True)#37201
    #train = model.predict_generator(train_generator,6910,verbose=True)#55273
    #train = model.predict_generator(train_generator,9127,verbose=True)#73014
    #train = model.predict_generator(train_generator,9123,verbose=True)#72981
    train = model.predict_generator(train_generator,11282,verbose=True)#90253
    test = model.predict_generator(test_generator, 1325,verbose=True)#10593
    
    #print(test .shape)
    with h5py.File(fileDir+"wh_code/cut/299_5p-incepV4.h5") as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)
        #h.create_dataset("val", data=val)
        #h.create_dataset("trainlabel", data=train_generator.classes)
        #h.create_dataset("vallabel", data=val_generator.classes)
   
    K.clear_session()
    '''

def xcep_2():
    input_tensor = Input((512, 512, 3))
    x = input_tensor
   
    x = Lambda(xception.preprocess_input)(x)
    base_model = Xception(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    batchsize=16
    image_size=(512,512)
    batch_X = np.zeros((batchsize,)+(512,512,3),dtype=K.floatx())
    train_feature=[]
    gen = ImageDataGenerator()
    train = gen.flow_from_directory("/your path/train", image_size, shuffle=False, batch_size=8)#18686
    postrain = '/your path/train_head/'
    for idx in range(0, len(train.filenames),batchsize):
        print(idx)
        if idx + batchsize<len(train.filenames):
            for i in range(batchsize):
                batch_X[i]=get_processed_image(postrain+train.filenames[idx+i].split('/')[1])
            train_feature.append(model.predict_on_batch(batch_X))
        else:
            length = len(train.filenames)-idx
            for i in range(length):
                batch_X[i]=get_processed_image(postrain+train.filenames[idx+i].split('/')[1])
            train_feature.append(model.predict_on_batch(batch_X[0:length]))

    train_feature = np.array(train_feature)
    train_feature = np.concatenate(train_feature, 0)
    '''
    test_feature=[]
    test = gen.flow_from_directory("/your path/test", image_size, shuffle=False, batch_size=8)#18686
    postest = '/your path/test_head/'
    for idx in range(0, len(test.filenames),batchsize):
        print(idx)
        if idx + batchsize<len(test.filenames):
            for i in range(batchsize):
                batch_X[i]=get_processed_image(postest+test.filenames[idx+i].split('/')[1])
            test_feature.append(model.predict_on_batch(batch_X))
        else:
            length = len(test.filenames)-idx
            for i in range(length):
                batch_X[i]=get_processed_image(postest+test.filenames[idx+i].split('/')[1])
            test_feature.append(model.predict_on_batch(batch_X[0:length]))

    test_feature = np.array(test_feature)
    test_feature = np.concatenate(test_feature, 0)

    print(train_feature.shape)
    print(test_feature.shape)

    '''
    with h5py.File('/your path/wh_code/train_head/512-xcep.h5', "w") as f:
         f.create_dataset("train", data=train_feature)
    #f.create_dataset("test", data=test_feature)
    #f.create_dataset("label", data=train.classes)
    


def dense161():
    weights_path = '/your path/wh_code/densenet/densenet161_weights_tf.h5'
    image_size=(224,224)
    base_model = densenet161.DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)
    #base_model = densenet161.densenet161_model(img_rows=224, img_cols=224, color_type=3)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    gen = ImageDataGenerator(preprocessing_function=densenet161.preprocess_input)
    print(base_model.output)
    #gen = ImageDataGenerator()
    #train_generator = gen.flow_from_directory("/your path/train_cut2p", image_size, shuffle=False, batch_size=8)#37189

    #train_generator = gen.flow_from_directory("/your path/traindata2_pre", image_size, shuffle=False, batch_size=8)#20227
    #val_generator = gen.flow_from_directory("/your path/val_cutpre", image_size, shuffle=False, batch_size=8)#3117
    #train_generator = gen.flow_from_directory("/your path/train", image_size, shuffle=False, batch_size=8)#18686
    

    #train_generator = gen.flow_from_directory("/your path/train_cut2p", image_size, shuffle=False, batch_size=8)#37189
    #train_generator = gen.flow_from_directory("/your path/train_cut", image_size, shuffle=False, batch_size=8)#18686
    #train_generator = gen.flow_from_directory("/your path/train",image_size, shuffle=False, batch_size=8)#18686
    #train_generator = gen.flow_from_directory("/your path/train_cut3p", image_size, shuffle=False, batch_size=8)#55298
    test_generator = gen.flow_from_directory("/your path/test", image_size, shuffle=False, batch_size=8, class_mode=None)#10593
    #test_generator = gen.flow_from_directory("/your path/test_cut", image_size, shuffle=False, batch_size=8, class_mode=None)#10593
    #train_generator = gen.flow_from_directory("/your path/train4p", image_size, shuffle=False, batch_size=8)#73049
    #train_generator = gen.flow_from_directory("/your path/train_cut4p", image_size, shuffle=False, batch_size=8)#72981
    #train_generator = gen.flow_from_directory("/your path/train_cut5p", image_size, shuffle=False, batch_size=8)#90253
    #train_generator = gen.flow_from_directory("/your path/train2p", image_size, shuffle=False, batch_size=8)#37201
    train_generator = gen.flow_from_directory("/your path/train3p", image_size, shuffle=False, batch_size=8)#55273
    #train = model.predict_generator(train_generator,2336,verbose=True)#18686
    #train = model.predict_generator(train_generator,4649,verbose=True)#37189
    #train = model.predict_generator(train_generator,6913,verbose=True)#55298
    #train = model.predict_generator(train_generator,6908,verbose=True)#55261
    #val = model.predict_generator(val_generator, 390,verbose=True)#3117
    #train = model.predict_generator(train_generator,4651,verbose=True)#37201
    train = model.predict_generator(train_generator,6910,verbose=True)#55273
    #train = model.predict_generator(train_generator,9127,verbose=True)#73014
    #train = model.predict_generator(train_generator,9123,verbose=True)#72981
    #train = model.predict_generator(train_generator,11282,verbose=True)#90253
    test = model.predict_generator(test_generator, 1325,verbose=True)#10593
    
    #print(test .shape)
    with h5py.File(fileDir+"wh_code/nocut/224_3p-dense161.h5") as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)
        #h.create_dataset("val", data=val)
        #h.create_dataset("trainlabel", data=train_generator.classes)
        #h.create_dataset("vallabel", data=val_generator.classes)
   
    K.clear_session()



def dense121():
    weights_path = '/your path/wh_code/densenet/densenet121_weights_tf.h5'
    image_size=(224,224)
    base_model = densenet121.DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)
    #base_model = densenet161.densenet161_model(img_rows=224, img_cols=224, color_type=3)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    gen = ImageDataGenerator(preprocessing_function=densenet121.preprocess_input)
    print(base_model.output)
    #gen = ImageDataGenerator()
    #train_generator = gen.flow_from_directory("/your path/train_cut2p", image_size, shuffle=False, batch_size=8)#37189

    #train_generator = gen.flow_from_directory("/your path/traindata2_pre", image_size, shuffle=False, batch_size=8)#20227
    #val_generator = gen.flow_from_directory("/your path/val_cutpre", image_size, shuffle=False, batch_size=8)#3117
    #train_generator = gen.flow_from_directory("/your path/train_cut", image_size, shuffle=False, batch_size=8)#18686
    

    #train_generator = gen.flow_from_directory("/your path/train_cut2p", image_size, shuffle=False, batch_size=8)#37189
    #train_generator = gen.flow_from_directory("/your path/train_cut", image_size, shuffle=False, batch_size=8)#18686
    #train_generator = gen.flow_from_directory("/your path/train",image_size, shuffle=False, batch_size=8)#18686
    #train_generator = gen.flow_from_directory("/your path/train_cut3p", image_size, shuffle=False, batch_size=8)#55298
    #test_generator = gen.flow_from_directory("/your path/test_cut", image_size, shuffle=False, batch_size=8, class_mode=None)#10593
    test_generator = gen.flow_from_directory("/your path/test", image_size, shuffle=False, batch_size=8, class_mode=None)#10593
    #train_generator = gen.flow_from_directory("/your path/train4p", image_size, shuffle=False, batch_size=8)#73049
    #train_generator = gen.flow_from_directory("/your path/train_cut4p", image_size, shuffle=False, batch_size=8)#72981
    #train_generator = gen.flow_from_directory("/your path/train_cut5p", image_size, shuffle=False, batch_size=8)#90253
    #train_generator = gen.flow_from_directory("/your path/train2p", image_size, shuffle=False, batch_size=8)#37201
    train_generator = gen.flow_from_directory("/your path/train3p", image_size, shuffle=False, batch_size=8)#55273
    #train = model.predict_generator(train_generator,2336,verbose=True)#18686
    #train = model.predict_generator(train_generator,4649,verbose=True)#37189
    #train = model.predict_generator(train_generator,6913,verbose=True)#55298
    #train = model.predict_generator(train_generator,6908,verbose=True)#55261
    #val = model.predict_generator(val_generator, 390,verbose=True)#3117
    #train = model.predict_generator(train_generator,4651,verbose=True)#37201
    train = model.predict_generator(train_generator,6910,verbose=True)#55273
    #train = model.predict_generator(train_generator,9127,verbose=True)#73014
    #train = model.predict_generator(train_generator,9123,verbose=True)#72981
    #train = model.predict_generator(train_generator,11282,verbose=True)#90253
    test = model.predict_generator(test_generator, 1325,verbose=True)#10593
    
    #print(test .shape)
    with h5py.File(fileDir+"wh_code/nocut/224_3p-dense121.h5") as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)
        #h.create_dataset("val", data=val)
        #h.create_dataset("trainlabel", data=train_generator.classes)
        #h.create_dataset("vallabel", data=val_generator.classes)
   
    K.clear_session()


def dense169():
    weights_path = '/your path/wh_code/densenet/densenet169_weights_tf.h5'
    image_size=(224,224)
    base_model = densenet169.DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)
    #base_model = densenet161.densenet161_model(img_rows=224, img_cols=224, color_type=3)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    gen = ImageDataGenerator(preprocessing_function=densenet169.preprocess_input)
    print(base_model.output)
    #gen = ImageDataGenerator()
    #train_generator = gen.flow_from_directory("/your path/train_cut2p", image_size, shuffle=False, batch_size=8)#37189
    #train_generator = gen.flow_from_directory("/your path/train_cut", image_size, shuffle=False, batch_size=8)#18686
    

    #train_generator = gen.flow_from_directory("/your path/train_cut2p", image_size, shuffle=False, batch_size=8)#37189
    #train_generator = gen.flow_from_directory("/your path/train_cut", image_size, shuffle=False, batch_size=8)#18686
    #train_generator = gen.flow_from_directory("/your path/train",image_size, shuffle=False, batch_size=8)#18686
    #train_generator = gen.flow_from_directory("/your path/train_cut3p", image_size, shuffle=False, batch_size=8)#55298
    #test_generator = gen.flow_from_directory("/your path/test_cut", image_size, shuffle=False, batch_size=8, class_mode=None)#10593
    test_generator = gen.flow_from_directory("/your path/test", image_size, shuffle=False, batch_size=8, class_mode=None)#10593
    #train_generator = gen.flow_from_directory("/your path/train4p", image_size, shuffle=False, batch_size=8)#73049
    #train_generator = gen.flow_from_directory("/your path/train_cut4p", image_size, shuffle=False, batch_size=8)#72981
    #train_generator = gen.flow_from_directory("/your path/train_cut5p", image_size, shuffle=False, batch_size=8)#90253
    #train_generator = gen.flow_from_directory("/your path/train2p", image_size, shuffle=False, batch_size=8)#37201
    train_generator = gen.flow_from_directory("/your path/train3p", image_size, shuffle=False, batch_size=8)#55273
    #train = model.predict_generator(train_generator,2336,verbose=True)#18686
    #train = model.predict_generator(train_generator,4649,verbose=True)#37189
    #train = model.predict_generator(train_generator,6913,verbose=True)#55298
    #train = model.predict_generator(train_generator,6908,verbose=True)#55261
    #val = model.predict_generator(val_generator, 390,verbose=True)#3117
    #train = model.predict_generator(train_generator,4651,verbose=True)#37201
    train = model.predict_generator(train_generator,6910,verbose=True)#55273
    #train = model.predict_generator(train_generator,9127,verbose=True)#73014
    #train = model.predict_generator(train_generator,9123,verbose=True)#72981
    #train = model.predict_generator(train_generator,11282,verbose=True)#90253
    test = model.predict_generator(test_generator, 1325,verbose=True)#10593
    
    #print(test .shape)
    with h5py.File(fileDir+"wh_code/nocut/224_3p-dense169.h5") as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)
        #h.create_dataset("val", data=val)
        #h.create_dataset("trainlabel", data=train_generator.classes)
        #h.create_dataset("vallabel", data=val_generator.classes)
   
    K.clear_session()


def res152():
    
    image_size=(448,448)
    base_model = resnet152.ResNet152(include_top=False, weights='imagenet')
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    print(model.input)
    gen = ImageDataGenerator()

    #train_generator = gen.flow_from_directory("/your path/train_cut2p", image_size, shuffle=False, batch_size=8)#37189

    #train_generator = gen.flow_from_directory("/your path/traindata2_pre", image_size, shuffle=False, batch_size=8)#20227
    #val_generator = gen.flow_from_directory("/your path/val_cutpre", image_size, shuffle=False, batch_size=8)#3117
    #train_generator = gen.flow_from_directory("/your path/train",image_size, shuffle=False, batch_size=8)#18686
    #train_generator = gen.flow_from_directory("/your path/train2p", image_size, shuffle=False, batch_size=8)#37175
    '''
    train_generator = gen.flow_from_directory("/your path/train_cut2p", image_size, shuffle=False, batch_size=8)#37178
    train = model.predict_generator(train_generator,4648,verbose=True)#37178
    with h5py.File(fileDir+"wh_code/train_cut/299_2p-res152.h5") as h:
        h.create_dataset("train", data=train)
    '''
    train_generator = gen.flow_from_directory("/your path/train_cut4p", image_size, shuffle=False, batch_size=8)#72897
    train = model.predict_generator(train_generator,9113,verbose=True)#72897
    with h5py.File(fileDir+"wh_code/train_cut/448_4p-res152.h5") as h:
        h.create_dataset("train", data=train)
    '''
    train_generator = gen.flow_from_directory("/your path/train3p_hun", image_size, shuffle=False, batch_size=8)#85828
    train = model.predict_generator(train_generator,10729,verbose=True)#85828
    with h5py.File(fileDir+"wh_code/train_nocut_hun/448_3p-res152.h5") as h:
        h.create_dataset("train", data=train)

    train_generator = gen.flow_from_directory("/your path/train_cut3p_hun", image_size, shuffle=False, batch_size=8)#85675
    train = model.predict_generator(train_generator,10710,verbose=True)#85675
    with h5py.File(fileDir+"wh_code/train_cut_hun/448_3p-res152.h5") as h:
        h.create_dataset("train", data=train)
    '''
   
    K.clear_session()
def res152_2():
    
    image_size=(448,448)
    base_model = resnet152.ResNet152(include_top=False, weights='imagenet')
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    print(model.input)

    batchsize=16
    batch_X = np.zeros((batchsize,)+(448,448,3),dtype=K.floatx())
    train_feature=[]
    gen = ImageDataGenerator()
    train = gen.flow_from_directory("/your path/train_hun", image_size, shuffle=False, batch_size=8)#18686
    postrain = '/your path/train_hun_head/'
    for idx in range(0, len(train.filenames),batchsize):
        print(idx)
        if idx + batchsize<len(train.filenames):
            for i in range(batchsize):
                batch_X[i]=get_processed_image(postrain+train.filenames[idx+i].split('/')[1])
            train_feature.append(model.predict_on_batch(batch_X))
        else:
            length = len(train.filenames)-idx
            for i in range(length):
                batch_X[i]=get_processed_image(postrain+train.filenames[idx+i].split('/')[1])
            train_feature.append(model.predict_on_batch(batch_X[0:length]))

    train_feature = np.array(train_feature)
    train_feature = np.concatenate(train_feature, 0)
    
    test_feature=[]
    test = gen.flow_from_directory("/your path/test", image_size, shuffle=False, batch_size=8)#18686
    postest = '/your path/test_head/'
    for idx in range(0, len(test.filenames),batchsize):
        print(idx)
        if idx + batchsize<len(test.filenames):
            for i in range(batchsize):
                batch_X[i]=get_processed_image(postest+test.filenames[idx+i].split('/')[1])
            test_feature.append(model.predict_on_batch(batch_X))
        else:
            length = len(test.filenames)-idx
            for i in range(length):
                batch_X[i]=get_processed_image(postest+test.filenames[idx+i].split('/')[1])
            test_feature.append(model.predict_on_batch(batch_X[0:length]))

    test_feature = np.array(test_feature)
    test_feature = np.concatenate(test_feature, 0)

    print(train_feature.shape)
    print(test_feature.shape)
    with h5py.File('/your path/wh_code/train_head_hun/448-res152.h5', "w") as f:
         f.create_dataset("train", data=train_feature)
    #f.create_dataset("test", data=test_feature)
    #f.create_dataset("label", data=train.classes)
    with h5py.File('/your path/wh_code/test_head/448-res152.h5', "w") as f:
         f.create_dataset("test", data=test_feature)

def main():
    #write_gap_train(VGG19,(512,512))
    #print("densenet161cut")
    #dense161()
    #dense121()
    #dense169()
    xcep_2()
    #write_gap_train(ResNet50,(512,512))
    #incepV4() #299
    #write_gap_train(Xception, (512,512), xception.preprocess_input) #229   
    #res152_2()
    
    #write_gap_test(InceptionV3, (1024,1024),inception_v3.preprocess_input,cho="non-random") #229
    #write_gap_test(Xception, (512,512), xception.preprocess_input,cho="non-random") #229
    #write_gap_train_myModel(InceptionV3,(512,512),inception_v3.preprocess_input)
if __name__ == '__main__':
    main()
