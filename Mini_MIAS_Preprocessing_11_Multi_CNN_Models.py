import os
import pandas as pd

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin')

from Mini_MIAS_1_Folders import Multiclass_Data_Model
from Mini_MIAS_1_Folders import Multiclass_Data_Model_Esp

from Mini_MIAS_8_CNN_Architectures import MobileNetV3Small_Pretrained
from Mini_MIAS_8_CNN_Architectures import MobileNetV3Large_Pretrained
from Mini_MIAS_8_CNN_Architectures import MobileNet_Pretrained

from Mini_MIAS_8_CNN_Architectures import ResNet50_PreTrained
from Mini_MIAS_8_CNN_Architectures import ResNet50V2_PreTrained
from Mini_MIAS_8_CNN_Architectures import ResNet152V2

from Mini_MIAS_8_CNN_Architectures import configuration_models

Model_Tested = ResNet50_PreTrained

NT = 'NT'
NO = 'NO'  
CLAHE = 'CLAHE'
HE = 'HE'
UM = 'UM'
CS = 'CS' 

def Testing_CNN_Models_Multiclass(Model, Images, Labels):

    # Parameters

    Images_Normal = Images[0]
    Images_Benign = Images[1]
    Images_Malignant = Images[2]

    Labels_Normal = Labels[0]
    Labels_Benign = Labels[1]
    Labels_Malignant = Labels[2]
    
    labels_Triclass = ['Normal', 'Benign', 'Malignant']

    X_size = 224
    Y_size = 224
    Valid_split = 0.1
    Epochs = 5

    df = pd.read_csv("D:\Mini-MIAS\Mini-MIAS Final\Multiclass_Data_CSV\Multiclass_DataFrame_MIAS_Data.csv")
    path = "D:\Mini-MIAS\Mini-MIAS Final\Multiclass_Data_CSV\Multiclass_DataFrame_MIAS_Data.csv"

    MainKeys = ['Model function', 'Technique', 'Labels', 'Xsize', 'Ysize', 'Validation split', 'Epochs','Images 1', 'Labels 1', 'Images 2', 'Labels 2']
    column_names = ['Name Model', "Model used", "Accuracy Training FE", "Accuracy Training LE", "Accuracy Testing", "Loss Train", "Loss Test", "Training images", "Test images", "Precision", "Recall", "F1_Score", "Time training", "Time testing", "Technique used", "TN", "FP", "FN", "TP", "Epochs", "Auc Normal", "Auc Benign", "Auc Malignant"]


    ModelValues = [Model, NT, labels_Triclass, X_size, Y_size, Valid_split, Epochs, Images_Normal, Labels_Normal, Images_Benign, Labels_Benign, Images_Malignant, Labels_Malignant]

    Score = configuration_models(MainKeys, ModelValues, Multiclass_Data_Model, Multiclass_Data_Model_Esp)

    update_csv_row(Score, df, column_names, path, 0)