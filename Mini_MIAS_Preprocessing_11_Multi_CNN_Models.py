import os
import pandas as pd

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin')

from Mini_MIAS_1_Folders import Multiclass_Data_Model
from Mini_MIAS_1_Folders import Multiclass_Data_Model_Esp

from Mini_MIAS_7_CNN_Architectures import PreTrainedModels

from Mini_MIAS_7_CNN_Architectures import MobileNetV3Small_Pretrained
from Mini_MIAS_7_CNN_Architectures import MobileNetV3Large_Pretrained
from Mini_MIAS_7_CNN_Architectures import MobileNet_Pretrained

from Mini_MIAS_7_CNN_Architectures import ResNet50_PreTrained
from Mini_MIAS_7_CNN_Architectures import ResNet50V2_PreTrained
from Mini_MIAS_7_CNN_Architectures import ResNet152V2

from Mini_MIAS_2_General_Functions import ConfigurationModels
from Mini_MIAS_2_General_Functions import UpdateCSV

Model_Tested = MobileNetV3Large_Pretrained

def Testing_CNN_Models_Multiclass(Model, Images, Labels):

    # Parameters

    Images_Normal = Images[0]
    Images_Benign = Images[1]
    Images_Malignant = Images[2]

    NOImages_Normal = Images[3]
    NOImages_Benign = Images[4]
    NOImages_Malignant = Images[5]

    Labels_Normal = Labels[0]
    Labels_Benign = Labels[1]
    Labels_Malignant = Labels[2]

    NOLabels_Normal = Labels[3]
    NOLabels_Benign = Labels[4]
    NOLabels_Malignant = Labels[4]
    
    labels_Triclass = ['Normal', 'Benign', 'Malignant']

    XsizeResized = 224
    YsizeResized = 224
    Valid_split = 0.1
    Digits_RV = 4
    Epochs = 5

    NT = 'NT'
    NO = 'NO'  
    CLAHE = 'CLAHE'
    HE = 'HE'
    UM = 'UM'
    CS = 'CS' 

    df = pd.read_csv("D:\Mini-MIAS\Mini-MIAS Final\Biclass_Data_CSV\Mutliclass_DataFrame_MIAS_Data.csv")
    path = "D:\Mini-MIAS\Mini-MIAS Final\Biclass_Data_CSV\Mutliclass_DataFrame_MIAS_Data.csv"

    MainKeys = ['Model function', 'Technique', 'Labels', 'Xsize', 'Ysize', 'Validation split', 'Epochs','Images 1', 'Labels 1', 'Images 2', 'Labels 2']
    column_names = ['Name Model', "Model used", "Accuracy Training FE", "Accuracy Training LE", "Accuracy Testing", "Loss Train", "Loss Test", "Training images", "Test images", "Precision", "Recall", "F1_Score", "Time training", "Time testing", "Technique used", "TN", "FP", "FN", "TP", "Epochs", "Auc Normal", "Auc Benign", "Auc Malignant"]

    ModelTest = MobileNetV3Large_Pretrained

    ModelValues =   [ModelTest, NT, labels_Triclass, XsizeResized, YsizeResized, Valid_split, Epochs, Images_Normal, Labels_Normal, Images_Benign, Labels_Benign, Images_Malignant, Labels_Malignant]
    ModelValues1 =  [ModelTest, NO, labels_Triclass, XsizeResized, YsizeResized, Valid_split, Epochs, NOImages_Normal, NOLabels_Normal, NOImages_Benign, NOLabels_Benign, NOImages_Malignant, NOLabels_Malignant]
    #ModelValues2 =  [ModelTest, CLAHE, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, CLAHEImages_Normal, CLAHELabels_Normal, CLAHEImages_Tumor, CLAHELabels_Tumor]
    #ModelValues3 =  [ModelTest, HE, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, HEImages_Normal, HELabels_Normal, HEImages_Tumor, HELabels_Tumor]
    #ModelValues4 =  [ModelTest, UM, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, UMImages_Normal, UMLabels_Normal, UMImages_Tumor, UMLabels_Tumor]
    #ModelValues5 =  [ModelTest, CS, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, CSImages_Normal, CSLabels_Normal, CSImages_Tumor, CSLabels_Tumor]
    
    ########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

    Score = ConfigurationModels(MainKeys, ModelValues, Multiclass_Data_Model, Multiclass_Data_Model_Esp)

    UpdateCSV(Score, df, column_names, path, 0)