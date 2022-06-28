import os
import pandas as pd

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin')

from Mini_MIAS_1_Folders import Biclass_Data_Model
from Mini_MIAS_1_Folders import Biclass_Data_Model_Esp

from Mini_MIAS_8_CNN_Architectures import PreTrainedModels

from Mini_MIAS_8_CNN_Architectures import MobileNetV3Small_Pretrained
from Mini_MIAS_8_CNN_Architectures import MobileNetV3Large_Pretrained
from Mini_MIAS_8_CNN_Architectures import MobileNet_Pretrained

from Mini_MIAS_8_CNN_Architectures import ResNet50_PreTrained
from Mini_MIAS_8_CNN_Architectures import ResNet50V2_PreTrained
from Mini_MIAS_8_CNN_Architectures import ResNet152V2

from Mini_MIAS_8_CNN_Architectures import configuration_models
from Mini_MIAS_8_CNN_Architectures import update_csv_row

Model_Tested = MobileNetV3Large_Pretrained

NT = 'NT'
NO = 'NO'  
CLAHE = 'CLAHE'
HE = 'HE'
UM = 'UM'
CS = 'CS' 

def Testing_CNN_Models_Biclass(Model, technique, Images, Labels):

    # Parameters

    Images_Normal = Images[0]
    Images_Tumor = Images[1]

    Labels_Normal = Labels[0]
    Labels_Tumor = Labels[1]
    
    labels_Biclass = ['Normal', 'Tumor']

    X_size = 224
    Y_size = 224
    Valid_split = 0.1
    Epochs = 5

    df = pd.read_csv("D:\Mini-MIAS\Mini-MIAS Final\Biclass_Data_CSV\Biclass_DataFrame_MIAS_Data.csv")
    path = "D:\Mini-MIAS\Mini-MIAS Final\Biclass_Data_CSV\Biclass_DataFrame_MIAS_Data.csv"

    dataframe_keys = ['Model function', 'Technique', 'Labels', 'Xsize', 'Ysize', 'Validation split', 'Epochs', 'Images 1', 'Labels 1', 'Images 2', 'Labels 2']
    column_names = ['Name Model', "Model used", "Accuracy Training FE", "Accuracy Training LE", "Accuracy Testing", "Loss Train", "Loss Test", "Training images", "Test images", "Precision", "Recall", "F1_Score", "Time training", "Time testing", "Technique used", "TN", "FP", "FN", "TP", "Epochs", "Auc"]

    parameters_model = [Model, technique, labels_Biclass, X_size, Y_size, Valid_split, Epochs, Images_Normal, Labels_Normal, Images_Tumor, Labels_Tumor]

    Score = configuration_models(dataframe_keys, parameters_model, Biclass_Data_Model, Biclass_Data_Model_Esp)

    update_csv_row(Score, df, column_names, path, 0)