from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

import h5py

# we don't have test data

fileDir = '/home/deeplearning/wh/baiduImage/'
def write_gap(MODEL, image_size, lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)

    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory(fileDir+"traindata2", image_size, shuffle=False, 
                                              batch_size=32)

    train = model.predict_generator(train_generator, 254,verbose=True)
    print(train .shape)
    
    with h5py.File(fileDir+"wh_code/2gap_%s.h5"%MODEL.__name__) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("label", data=train_generator.classes)
def main():  
    write_gap(Xception, (512, 512), xception.preprocess_input)
if __name__ == '__main__':
    main()
