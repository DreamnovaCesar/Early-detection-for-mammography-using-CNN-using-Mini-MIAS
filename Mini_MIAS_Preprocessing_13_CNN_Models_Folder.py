import os
import pandas as pd
import tensorflow as tf

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin')

from Mini_MIAS_1_Folders import Biclass_Data_CSV
from Mini_MIAS_1_Folders import Biclass_Data_Model
from Mini_MIAS_1_Folders import Biclass_Data_Model_Esp

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

from Mini_MIAS_8_CNN_Architectures import configuration_models_folder

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

def training_testing_validation_from_directory(Models, Folder, Technique):

    # * Parameters
    #Labels_biclass = ['Normal', 'Tumor']
    Labels_triclass = ['Normal', 'Benign', 'Malignant']
    X_size = 224
    Y_size = 224
    Epochs = 5
    Valid_split = 0.1

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

    # * Lists
    Column_names = ['Name Model', "Model used", "Accuracy Training FE", "Accuracy Training LE", "Accuracy Testing", "Loss Train", "Loss Test", "Training images", "Test images", "Precision", "Recall", "F1_Score", "Time training", "Time testing", "Technique used", "TN", "FP", "FN", "TP", "Epochs", "Auc"]
    #Dataframe_keys = ['Model function', 'Technique', 'Labels', 'Xsize', 'Ysize', 'Validation split', 'Epochs', 'Images 1', 'Labels 1', 'Images 2', 'Labels 2']
    
    Dataframe_save_mias = pd.DataFrame(columns = Column_names)
    
    #for Index, model in enumerate(Model):
        #Model_function, Model_name, Model_name_letters = model(X_size, Y_size, len(Labels_biclass))

    # * Save dataframe in the folder given
    #Dataframe_save_mias_name = 'Biclass_' + 'Dataframe_' + 'CNN_' + str(Technique) + '_' + str(Model_name_letters) + '.csv'
    Dataframe_save_mias_name = 'Biclass_' + 'Dataframe_' + 'CNN_' + 'Folder' + str(Technique) + '.csv'
    Dataframe_save_mias_folder = os.path.join(Biclass_Data_CSV, Dataframe_save_mias_name)

    Dataframe_save_mias.to_csv(Dataframe_save_mias_folder)
    Dataframe_save_mias = pd.read_csv(Dataframe_save_mias_folder)

    #print(Dataframe_save_mias_folder)

    Info_dataframe = configuration_models_folder(train_generator, valid_generator, test_generator, Dataframe_save_mias, Dataframe_save_mias_folder, Models, Technique, Labels_triclass, Column_names, X_size, Y_size, Valid_split, Epochs, Biclass_Data_Model, Biclass_Data_Model_Esp)

    return Info_dataframe
