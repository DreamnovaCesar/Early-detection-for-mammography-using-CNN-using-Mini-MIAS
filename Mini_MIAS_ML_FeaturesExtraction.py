import os
import pandas as pd

from Mini_MIAS_1_Folders import Biclass_Data_CSV
from Mini_MIAS_1_Folders import Biclass_Data_Model
from Mini_MIAS_1_Folders import Biclass_Data_Model_Esp

from Mini_MIAS_1_Folders import Multiclass_Data_CSV
from Mini_MIAS_1_Folders import Multiclass_Data_Model
from Mini_MIAS_1_Folders import Multiclass_Data_Model_Esp

from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Normal
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Tumor

from Mini_MIAS_ML_Functions import SVM
from Mini_MIAS_ML_Functions import MLP
from Mini_MIAS_ML_Functions import KNN
from Mini_MIAS_ML_Functions import RF
from Mini_MIAS_ML_Functions import DT
from Mini_MIAS_ML_Functions import GBC

from Mini_MIAS_7_Extract_Feature import featureExtraction

from Mini_MIAS_2_General_Functions import concat_dataframe

from Mini_MIAS_ML_Functions import Machine_learning_config
from Mini_MIAS_ML_Functions import Overwrite_row_CSV


def Testing_ML_Models_Biclass_FOF(Model, Technique, All_images, All_labels):

    Column_names = ["Model name", "Model", "Accuracy", "Precision", "Recall", "F1 Score", "Training images", "Test images", "Time training", "Technique", "TN", "FP", "FN", "TP", "AUC"]
    Dataframe_save_mias = pd.DataFrame(columns = Column_names)

    # * Save dataframe in the folder given
    Dataframe_save_mias_name = 'Biclass' + '_Dataframe_' + str(Technique)  + '.csv'
    Dataframe_save_mias_folder = os.path.join(Biclass_Data_CSV, Dataframe_save_mias_name)

    Dataframe_save_mias.to_csv(Dataframe_save_mias_folder)
    Dataframe_save_mias = pd.read_csv(Dataframe_save_mias_folder)

    print(Dataframe_save_mias_folder)

    Labels_biclass = ['Normal', 'Tumor']
    #Labels_triclass = ['Normal', 'Benign', 'Malignant']

    Images_Normal = All_images[0]
    Images_Tumor = All_images[1]

    Labels_Normal = All_labels[0]
    Labels_Tumor = All_labels[1]
        
    ML_extraction_biclass_normal = featureExtraction(Images = Images_Normal, Label = Labels_Normal)
    ML_extraction_biclass_tumor = featureExtraction(Images = Images_Tumor, Label = Labels_Tumor)
    
    Dataframe_normal, X_normal, Y_normal, Technique_name_normal = ML_extraction_biclass_normal.textures_Feature_first_order_from_images()
    Dataframe_tumor, X_tumor, Y_tumor, Technique_name_tumor = ML_extraction_biclass_tumor.textures_Feature_first_order_from_images()
    Dataframe_concat = concat_dataframe(Dataframe_normal, Dataframe_tumor, Folder = Biclass_Data_CSV, Class = 'Biclass', Technique = Technique)

    Machine_learning_config(Dataframe_concat, Dataframe_save_mias, Dataframe_save_mias_folder, Column_names, Model, Technique, Labels_biclass, Biclass_Data_CSV, Biclass_Data_Model, Technique_name_normal)

    #ML_extraction_biclass.textures_Feature_GLCM_from_images(Images_Normal, Labels_Normal)
    #ML_extraction_biclass.textures_Feature_GLCM_from_images(Images_Tumor, Labels_Tumor)

    #ML_extraction_biclass.textures_Feature_GLRLM_from_images(Images_Normal, Labels_Normal)
    #ML_extraction_biclass.textures_Feature_GLRLM_from_images(Images_Tumor, Labels_Tumor)

def Testing_ML_Models_Multiclass_FOF(Model, Technique, All_images, ALL_labels):


    Column_names = ["Model name", "Model", "Accuracy", "Precision", "Recall", "F1_Score", "Training images", "Test images", "Time training", "Technique", "TN", "FP", "FN", "TP", "Auc Normal", "Auc Benign", "Auc Malignant"]
    Dataframe_save_mias = pd.DataFrame(columns = Column_names)

    # * Save dataframe in the folder given
    Dataframe_save_mias_name = 'Multiclass' + '_Dataframe_' + str(Technique)  + '.csv'
    Dataframe_save_mias_folder = os.path.join(Multiclass_Data_CSV, Dataframe_save_mias_name)

    Dataframe_save_mias.to_csv(Dataframe_save_mias_folder)
    Dataframe_save_mias = pd.read_csv(Dataframe_save_mias_folder)

    print(Dataframe_save_mias_folder)

    #Labels_biclass = ['Normal', 'Tumor']
    Labels_triclass = ['Normal', 'Benign', 'Malignant']

    Images_Normal = All_images[0]
    Images_benign = All_images[1]
    Images_malignant = All_images[2]

    Labels_Normal = ALL_labels[0]
    Labels_benign = ALL_labels[1]
    Labels_malignant = ALL_labels[2]
        
    ML_extraction_biclass_normal = featureExtraction(Images = Images_Normal, Label = Labels_Normal)
    ML_extraction_biclass_benign = featureExtraction(Images = Images_benign, Label = Labels_benign)
    ML_extraction_biclass_malignant = featureExtraction(Images = Images_malignant, Label = Labels_malignant)
    
    Dataframe_normal, X_normal, Y_normal, Technique_name_normal = ML_extraction_biclass_normal.textures_Feature_first_order_from_images()
    Dataframe_benign, X_benign, Y_benign, Technique_name_benign = ML_extraction_biclass_benign.textures_Feature_first_order_from_images()
    Dataframe_malignant, X_malignant, Y_malignant, Technique_name_malignant = ML_extraction_biclass_malignant.textures_Feature_first_order_from_images()

    Dataframe_concat = concat_dataframe(Dataframe_normal, Dataframe_benign, Dataframe_malignant, Folder = Multiclass_Data_CSV, Class = 'Multiclass', Technique = Technique)

    Machine_learning_config(Dataframe_concat, Dataframe_save_mias, Dataframe_save_mias_folder, Column_names, Model, Technique, Labels_triclass, Multiclass_Data_CSV, Multiclass_Data_Model, Technique_name_normal)


    #ML_extraction_biclass.textures_Feature_GLCM_from_images(Images_Normal, Labels_Normal)
    #ML_extraction_biclass.textures_Feature_GLCM_from_images(Images_Tumor, Labels_Tumor)

    #ML_extraction_biclass.textures_Feature_GLRLM_from_images(Images_Normal, Labels_Normal)
    #ML_extraction_biclass.textures_Feature_GLRLM_from_images(Images_Tumor, Labels_Tumor)
