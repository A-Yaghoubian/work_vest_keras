import tensorflow as tf
from google.colab import drive
from google.colab import files

drive.mount('/content/drive')

import os
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.optimizers import SGD
from keras.layers import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Input, Reshape, Concatenate, GlobalAveragePooling2D, BatchNormalization, Dropout, Activation, GlobalMaxPooling2D
from tensorflow.keras.utils import Sequence

% cp /content/drive/MyDrive/clean_data.zip /content/clean_data.zip

# ! unzip clean_data.zip

os.path.abspath('')

%cd /content

data = glob('clean_data/*/*.jpg')
print(data)

labels = []
images = []
for path in tqdm(data):
    img = Image.open(path).convert('RGB').resize((64, 64))
    img = np.array(img) / 255.0
    images.append(img)
    label = path.split('/')[-2]
    labels.append(int(label))

labels = np.array(labels)
images = np.array(images)
print(labels.shape, images.shape)


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    rescale=(1/255))

# datagen.fit(x_train)

generator = datagen.flow_from_directory("clean_data/", target_size=(64, 64), class_mode='binary')
validation_gen = datagen.flow_from_directory("clean_data/", target_size=(64, 64), class_mode='binary', subset='validation')



X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.33, random_state=42)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Input(shape=(64,64,3)))
# model.add(tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='SAME', activation='relu')) 
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
# model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation="relu"))
# model.add(tf.keras.layers.Dropout(0.25))
# model.add(tf.keras.layers.Dense(64, activation="relu"))
# model.add(tf.keras.layers.Dense(2, activation="softmax"))

input_1 = tf.keras.layers.Input(shape=(64,64,3))
c2d_1 = tf.keras.layers.Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='SAME', activation='relu')(input_1)
c2d_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c2d_1)

c2d_3 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(input_1)
c2d_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c2d_3)

c2d_5 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='SAME', activation='relu')(input_1)
c2d_5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c2d_5)

# concat = tf.keras.layers.Concatenate()([c2d_1, c2d_3, c2d_5])
added = tf.keras.layers.Add()([c2d_1, c2d_3, c2d_5])

x = (tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='SAME', activation='relu'))(added)
x = (tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))(x)
x = (tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu'))(x)
x = (tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))(x)
x = (tf.keras.layers.Flatten())(x)
# x = (tf.keras.layers.Dropout(0.25))(x)
x = (tf.keras.layers.Dense(128, activation="relu"))(x)
x = (tf.keras.layers.Dense(64, activation="relu"))(x)
x = (tf.keras.layers.Dense(2, activation="softmax"))(x)

model = tf.keras.Model(input_1, x)

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9), #^check the lr and momentum
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit_generator(generator, epochs=50, validation_data=validation_gen)

model.summary()

model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_test, y_test))
