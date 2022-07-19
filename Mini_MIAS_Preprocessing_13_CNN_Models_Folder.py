
import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import Xception

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

src_path_train = r"D:\Mini-MIAS\Mini-MIAS Final\Mini_MIAS_NO_Cropped_Images_Biclass_Split\train"
src_path_val = "D:/Mini-MIAS/Mini-MIAS Final/Mini_MIAS_NO_Cropped_Images_Biclass_Split/val"
src_path_test = r"D:\Mini-MIAS\Mini-MIAS Final\Mini_MIAS_NO_Cropped_Images_Biclass_Split\test"

train_datagen = ImageDataGenerator()

val_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

batch_size = 32

train_generator = train_datagen.flow_from_directory(
    directory=src_path_train,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
valid_generator = val_datagen.flow_from_directory(
    directory=src_path_val,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=32
)
test_generator = test_datagen.flow_from_directory(
    directory=src_path_test,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode=None,
    shuffle=False,
    seed=42
)

def prepare_model1():
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
    return model

def prepare_model():
    conv_base = ResNet50(weights='imagenet', include_top = False, input_shape=(224, 224, 3))
    model = Sequential()
    model.add(conv_base)

    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(2, activation='sigmoid'))

    conv_base.trainable = False
    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
    return model

model = prepare_model1()
model.fit_generator(  train_generator,
            validation_data = train_generator,
            steps_per_epoch = train_generator.n//train_generator.batch_size,
            validation_steps = valid_generator.n//valid_generator.batch_size,
            epochs = 8)

score = model.evaluate(valid_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])