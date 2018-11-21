

import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Reshape, Flatten, Dropout, Activation
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from imageio import imread
from imgaug import augmenters as iaa


def galaxy_cnn(input_size, output_size, drop_out=False, batch_norm=False, dense_size=1024):
    '''
    Network architecture
    input_size: Input size (shape of the input image)
    output_size: Output values to predict
    drop_out: Activate drop out in the architecture in the fully connected layer
    batch_normalization: Activate batch_normalization between the convolutional layers
    dense_size: number of nodes in the fully connected layers
    return: keras model
    '''
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_size))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(dense_size, activation='relu'))
    if drop_out:
        model.add(Dropout(0.5))
    model.add(Dense(dense_size, activation='relu'))
    if drop_out:
        model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='sigmoid'))
    return model

def galaxy_cnn_2(input_size, output_size):
    '''
    Network architecture
    input_size: Input size (shape of the input image)
    output_size: Output values to predict
    drop_out: Activate drop out in the architecture in the fully connected layer
    batch_normalization: Activate batch_normalization between the convolutional layers
    dense_size: number of nodes in the fully connected layers
    return: keras model
    '''
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_size))
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=(3, 3)))
    model.add(Conv2D(128, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=(3, 3)))
    model.add(Conv2D(256, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(3, 3)))
    model.add(Conv2D(512, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation=None))
    return model

def preprocess_input(np_img):
    '''
    Preprocess image and make transformations
    np_img: image to transform
    return: transform image
    '''
    # Normilize image
    np_img = np_img / 255
    
    # randome data augmentation
    aug = np.random.choice([0, 1])
    if aug == True:
        # Random horizontal and vertical flip
        seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.5)])
        np_img = seq.augment_image(np_img)
        list_zoom = [1.4, 1.3, 1.2, 1.5]
        # Zoom out randomly
        if np.random.choice([True, False]):
            random_x_f = np.random.choice(list_zoom)
            random_x_s = np.random.choice(list_zoom)
            random_y_f = np.random.choice(list_zoom)
            random_y_s = np.random.choice(list_zoom)
            seq = iaa.Sequential([iaa.Affine(scale={"x": (random_x_f, random_x_s), "y": (random_y_f, random_y_s)})])
            np_img = seq.augment_image(np_img)
        
    return np_img

def generator(path_images, labels, batch_size, val=False):
    '''
    Generate batches of images
    path_images: where the images are located
    labels: labels in a numpy array
    batch_size: batch_size to load into the network
    val: If the data generator will be used for validation then data augmentation is not applied
    return: batch of images and labels
    '''
    count = 0
    while True:
        batch_features = []
        batch_labels = []
        for i in range(batch_size):
            # Read image
            name_img = labels[count][0] + '.jpg'
            img = imread(path_images + '/' + name_img)
            
            # Image preprocessing
            if val:
                img = img / 255
            else:
                img = preprocess_input(img)

            label = labels[count][1:]

            batch_features.append(img)
            batch_labels.append(label)
            
            # Restart counter when it has reached the size 
            # of the data set
            if count == labels.shape[0] - 1:
                count = 0
                break
            count += 1

            
        yield np.array(batch_features), np.array(batch_labels)

def generator_predictions(path_images, labels, batch_size):
    '''
    Generate batches of images
    path_images: where the images are located
    labels: labels in a numpy array
    batch_size: batch_size to load into the network
    val: If the data generator will be used for validation then data augmentation is not applied
    return: batch of images
    '''
    count = 0
    while True:
        batch_features = []
        for i in range(batch_size):
            # Read image
            name_img = labels[count]
            img = imread(path_images + '/' + name_img)
            
            # Image preprocessing
            img = img / 255

            batch_features.append(img)
            
            # Restart counter when it has reached the size 
            # of the data set
            if count == labels.shape[0] - 1:
                count = 0
                break
            count += 1

            
        yield np.array(batch_features)