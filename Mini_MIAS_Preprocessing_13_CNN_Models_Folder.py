
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

def training_testing_validation_from_directory(Folder):

    #Name_dir = os.path.dirname(Folder)
    #Name_base = os.path.basename(Folder)

    batch_size = 32

    X_size = 224
    Y_size = 224

    Shape = (X_size, Y_size)

    Name_folder_training = Folder + '/' + 'train'
    Name_folder_val = Folder + '/' + 'val'
    Name_folder_test = Folder + '/' + 'test'

    train_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        directory = Name_folder_training,
        target_size = Shape,
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = True,
        seed = 42
    )

    valid_generator = val_datagen.flow_from_directory(
        directory = Name_folder_val,
        target_size = Shape,
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = True,
        seed = 32        
    )

    test_generator = test_datagen.flow_from_directory(
        directory = Name_folder_test,
        target_size = Shape,
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = False,
        seed = 42
    )

    def prepare_model1():
        model = Sequential()
        model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (224, 224, 3)))
        model.add(AveragePooling2D(pool_size = (2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
        model.add(AveragePooling2D(pool_size = (2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
        model.add(AveragePooling2D(pool_size = (2, 2)))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(32, activation = 'relu'))
        model.add(Dense(3, activation = 'softmax'))
        model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ['accuracy'])
        return model


    def prepare_model():
        conv_base = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))
        model = Sequential()
        model.add(conv_base)

        model.add(Flatten())
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(3, activation = 'softmax'))

        conv_base.trainable = False
        model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ['accuracy'])
        return model

    model = prepare_model1()
    model.fit(  train_generator,
                validation_data = train_generator,
                steps_per_epoch = train_generator.n//train_generator.batch_size,
                validation_steps = valid_generator.n//valid_generator.batch_size,
                epochs = 8)

    score = model.evaluate(test_generator)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


