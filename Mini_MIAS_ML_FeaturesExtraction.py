import pandas as pd

from Mini_MIAS_1_Folders import Biclass_Data_CSV
from Mini_MIAS_1_Folders import Biclass_Data_Model
from Mini_MIAS_1_Folders import Biclass_Data_Model_Esp

from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Normal
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Tumor

from Mini_MIAS_ML_Functions import SVM
from Mini_MIAS_ML_Functions import MLP
from Mini_MIAS_ML_Functions import KNN
from Mini_MIAS_ML_Functions import RF
from Mini_MIAS_ML_Functions import DT
from Mini_MIAS_ML_Functions import GBC

from Mini_MIAS_7_Extract_Feature import featureExtraction
from Mini_MIAS_ML_Functions import MLConfigurationModels


def Testing_CNN_Models_Biclass_ML(Model, technique, Images, Labels):

    Dataframe = pd.read_csv("D:\Mini-MIAS\MIAS VS\DataCSV\DataFrame_Binary_MIAS_Data_ML_FO.csv")
    #path = 'D:\Mini-MIAS\MIAS VS\DataCSV\DataFrame_Binary_MIAS_Data_ML_FO.csv'
    #MainKeys = ['Model function', 'Technique', 'Labels','Images 1', 'Labels 1', 'Images 2', 'Labels 2']

    Keys_biclass = ['Images 1', 'Labels 1', 'Images 2', 'Labels 2']
    Keys_multiclass = ['Images 1', 'Labels 1', 'Images 2', 'Labels 2', 'Images 3', 'Labels 3']

    Column_names = ["Model used", "Accuracy", "Precision", "Recall", "F1_Score", "Training images", "Test images", "Time training", "Technique", "TN", "FP", "FN", "TP", "AUC"]

    Labels_biclass = ['Normal', 'Tumor']
    Labels_triclass = ['Normal', 'Benign', 'Malignant']

    NT = 'NT'
    NO = 'NO'
    CLAHE = 'CLAHE'
    HE = 'HE'
    UM = 'UM'
    CS = 'CS'

    Images_Normal = Images[0]
    Images_Tumor = Images[1]

    Labels_Normal = Labels[0]
    Labels_Tumor = Labels[1]
        

    ML_extraction_biclass = featureExtraction()

    Dataframe, X, Y, Technique_name = ML_extraction_biclass.textures_Feature_first_order_from_images(Images_Normal, Labels_Normal)
    ML_extraction_biclass.textures_Feature_first_order_from_images(Images_Tumor, Labels_Tumor)

    ML_extraction_biclass.textures_Feature_GLCM_from_images(Images_Normal, Labels_Normal)
    ML_extraction_biclass.textures_Feature_GLCM_from_images(Images_Tumor, Labels_Tumor)

    ML_extraction_biclass.textures_Feature_GLRLM_from_images(Images_Normal, Labels_Normal)
    ML_extraction_biclass.textures_Feature_GLRLM_from_images(Images_Tumor, Labels_Tumor)