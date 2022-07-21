import numpy as np

from Mini_MIAS_2_General_Functions import concat_dataframe

from Mini_MIAS_2_General_Functions import split_folders_train_test_val
from Mini_MIAS_Preprocessing_1_ChangeExtension import preprocessing_ChangeFormat

from Mini_MIAS_Preprocessing_2_Cropped_MIAS_Mammograms import preprocessing_Cropped_MIAS_Mammograms

from Mini_MIAS_Preprocessing_3_Kmeans_GLCM import preprocessing_Kmeans_GLCM_Tumor
from Mini_MIAS_Preprocessing_3_Kmeans_GLCM_Mal_Ben import preprocessing_Kmeans_GLCM_Benign
from Mini_MIAS_Preprocessing_3_Kmeans_GLCM_Mal_Ben import preprocessing_Kmeans_GLCM_Malignant

from Mini_MIAS_Preprocessing_4_Resize import preprocessing_Resize

from Mini_MIAS_Preprocessing_5_Select_technique import preprocessing_technique_Biclass
from Mini_MIAS_Preprocessing_5_Select_technique import preprocessing_technique_Multiclass

from Mini_MIAS_Preprocessing_10_Data_Augmentation import preprocessing_DataAugmentation_Biclass
from Mini_MIAS_Preprocessing_10_Multi_Data_Augmentation import preprocessing_DataAugmentation_Multiclass

from Mini_MIAS_Preprocessing_11_CNN_Models import Testing_CNN_Models_Biclass
from Mini_MIAS_Preprocessing_11_Multi_CNN_Models import Testing_CNN_Models_Multiclass

from Mini_MIAS_Preprocessing_12_Data_Augmentation_Folder import Split_Folders_Each_Technique
from Mini_MIAS_Preprocessing_13_CNN_Models_Folder import training_testing_validation_from_directory

from Mini_MIAS_8_CNN_Architectures import MobileNet_Pretrained
from Mini_MIAS_8_CNN_Architectures import MobileNetV3Small_Pretrained
from Mini_MIAS_8_CNN_Architectures import MobileNetV3Large_Pretrained

from Mini_MIAS_8_CNN_Architectures import ResNet50_PreTrained
from Mini_MIAS_8_CNN_Architectures import ResNet152_PreTrained
from Mini_MIAS_8_CNN_Architectures import ResNet152V2_PreTrained

from Mini_MIAS_8_CNN_Architectures import ResNet50_pretrained_folder
from Mini_MIAS_8_CNN_Architectures import MobileNet_pretrained_folder

from Mini_MIAS_ML_FeaturesExtraction import Testing_ML_Models_Biclass_FOF
from Mini_MIAS_ML_FeaturesExtraction import Testing_ML_Models_Multiclass_FOF

from Mini_MIAS_ML_Functions import SVM
from Mini_MIAS_ML_Functions import Multi_SVM
from Mini_MIAS_ML_Functions import MLP
from Mini_MIAS_ML_Functions import KNN
from Mini_MIAS_ML_Functions import RF
from Mini_MIAS_ML_Functions import DT
from Mini_MIAS_ML_Functions import GBC

from Mini_MIAS_1_Folders import Biclass_Data_CSV
from Mini_MIAS_1_Folders import Multiclass_Data_CSV

from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Biclass 
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Multiclass 
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Normal
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Tumor
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Benign
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Malignant

from Mini_MIAS_1_Folders import Mini_MIAS_NO_Cropped_Images_Biclass
from Mini_MIAS_1_Folders import Mini_MIAS_NO_Cropped_Images_Multiclass 
from Mini_MIAS_1_Folders import Mini_MIAS_NO_Cropped_Images_Normal
from Mini_MIAS_1_Folders import Mini_MIAS_NO_Cropped_Images_Tumor
from Mini_MIAS_1_Folders import Mini_MIAS_NO_Cropped_Images_Benign
from Mini_MIAS_1_Folders import Mini_MIAS_NO_Cropped_Images_Malignant

from Mini_MIAS_1_Folders import Mini_MIAS_CLAHE_Cropped_Images_Biclass
from Mini_MIAS_1_Folders import Mini_MIAS_CLAHE_Cropped_Images_Multiclass 
from Mini_MIAS_1_Folders import Mini_MIAS_CLAHE_Cropped_Images_Normal
from Mini_MIAS_1_Folders import Mini_MIAS_CLAHE_Cropped_Images_Tumor
from Mini_MIAS_1_Folders import Mini_MIAS_CLAHE_Cropped_Images_Benign
from Mini_MIAS_1_Folders import Mini_MIAS_CLAHE_Cropped_Images_Malignant

from Mini_MIAS_1_Folders import Mini_MIAS_HE_Cropped_Images_Biclass
from Mini_MIAS_1_Folders import Mini_MIAS_HE_Cropped_Images_Multiclass 
from Mini_MIAS_1_Folders import Mini_MIAS_HE_Cropped_Images_Normal
from Mini_MIAS_1_Folders import Mini_MIAS_HE_Cropped_Images_Tumor
from Mini_MIAS_1_Folders import Mini_MIAS_HE_Cropped_Images_Benign
from Mini_MIAS_1_Folders import Mini_MIAS_HE_Cropped_Images_Malignant

from Mini_MIAS_1_Folders import Mini_MIAS_UM_Cropped_Images_Biclass
from Mini_MIAS_1_Folders import Mini_MIAS_UM_Cropped_Images_Multiclass 
from Mini_MIAS_1_Folders import Mini_MIAS_UM_Cropped_Images_Normal
from Mini_MIAS_1_Folders import Mini_MIAS_UM_Cropped_Images_Tumor
from Mini_MIAS_1_Folders import Mini_MIAS_UM_Cropped_Images_Benign
from Mini_MIAS_1_Folders import Mini_MIAS_UM_Cropped_Images_Malignant

from Mini_MIAS_1_Folders import Mini_MIAS_CS_Cropped_Images_Biclass
from Mini_MIAS_1_Folders import Mini_MIAS_CS_Cropped_Images_Multiclass 
from Mini_MIAS_1_Folders import Mini_MIAS_CS_Cropped_Images_Normal
from Mini_MIAS_1_Folders import Mini_MIAS_CS_Cropped_Images_Tumor
from Mini_MIAS_1_Folders import Mini_MIAS_CS_Cropped_Images_Benign
from Mini_MIAS_1_Folders import Mini_MIAS_CS_Cropped_Images_Malignant

def main():

    PGM = ".pgm"
    PNG = ".png"
    TIFF = ".tiff"

    Model_CNN = (MobileNet_Pretrained, MobileNetV3Small_Pretrained, MobileNetV3Large_Pretrained)
    Model_CNN_R = (ResNet50_PreTrained, ResNet152_PreTrained)
    Model_ML = (Multi_SVM, MLP, KNN, RF, DT, GBC)
    
    Model_CNN_test = (MobileNet_pretrained_folder, ResNet50_pretrained_folder)

    #preprocessing_ChangeFormat(PGM, PNG)
    #preprocessing_ChangeFormat(PGM, TIFF)
    #preprocessing_Cropped_MIAS_Mammograms()
    #preprocessing_Kmeans_GLCM_Tumor()
    #preprocessing_Kmeans_GLCM_Benign()
    #preprocessing_Kmeans_GLCM_Malignant()

    #preprocessing_Resize(Mini_MIAS_NT_Cropped_Images_Normal, Mini_MIAS_NT_Cropped_Images_Tumor, Mini_MIAS_NT_Cropped_Images_Benign, Mini_MIAS_NT_Cropped_Images_Malignant)
    
    """

    preprocessing_technique_Biclass('NO', Mini_MIAS_NT_Cropped_Images_Normal, Mini_MIAS_NT_Cropped_Images_Tumor, Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Tumor)
    preprocessing_technique_Biclass('CLAHE', Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Tumor, Mini_MIAS_CLAHE_Cropped_Images_Normal, Mini_MIAS_CLAHE_Cropped_Images_Tumor)
    preprocessing_technique_Biclass('HE', Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Tumor, Mini_MIAS_HE_Cropped_Images_Normal, Mini_MIAS_HE_Cropped_Images_Tumor)
    preprocessing_technique_Biclass('UM', Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Tumor, Mini_MIAS_UM_Cropped_Images_Normal, Mini_MIAS_UM_Cropped_Images_Tumor)
    preprocessing_technique_Biclass('CS', Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Tumor, Mini_MIAS_CS_Cropped_Images_Normal, Mini_MIAS_CS_Cropped_Images_Tumor)
    
    """

    """

    preprocessing_technique_Multiclass( 'NO', Mini_MIAS_NT_Cropped_Images_Normal, Mini_MIAS_NT_Cropped_Images_Benign, Mini_MIAS_NT_Cropped_Images_Malignant, 
                                            Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Benign, Mini_MIAS_NO_Cropped_Images_Malignant)

    preprocessing_technique_Multiclass( 'CLAHE', Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Benign, Mini_MIAS_NO_Cropped_Images_Malignant, 
                                            Mini_MIAS_CLAHE_Cropped_Images_Normal, Mini_MIAS_CLAHE_Cropped_Images_Benign, Mini_MIAS_CLAHE_Cropped_Images_Malignant)

    preprocessing_technique_Multiclass( 'HE', Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Benign, Mini_MIAS_NO_Cropped_Images_Malignant, 
                                            Mini_MIAS_HE_Cropped_Images_Normal, Mini_MIAS_HE_Cropped_Images_Benign, Mini_MIAS_HE_Cropped_Images_Malignant)

    preprocessing_technique_Multiclass( 'UM', Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Benign, Mini_MIAS_NO_Cropped_Images_Malignant, 
                                            Mini_MIAS_UM_Cropped_Images_Normal, Mini_MIAS_UM_Cropped_Images_Benign, Mini_MIAS_UM_Cropped_Images_Malignant)

    preprocessing_technique_Multiclass( 'CS', Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Benign, Mini_MIAS_NO_Cropped_Images_Malignant, 
                                            Mini_MIAS_CS_Cropped_Images_Normal, Mini_MIAS_CS_Cropped_Images_Benign, Mini_MIAS_CS_Cropped_Images_Malignant)
    
    """

    #"""

    #Images, Labels = preprocessing_DataAugmentation_Biclass(Mini_MIAS_NT_Cropped_Images_Normal, Mini_MIAS_NT_Cropped_Images_Tumor, Mini_MIAS_NT_Cropped_Images_Biclass)
    #Dataframe_final_NT = Testing_CNN_Models_Biclass(Model_CNN, 'NT', Images, Labels)

    #Images, Labels = preprocessing_DataAugmentation_Biclass(Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Tumor, Mini_MIAS_NO_Cropped_Images_Biclass)
    #Dataframe_final_NO = Testing_CNN_Models_Biclass(Model_CNN, 'NO', Images, Labels)

    #Images, Labels = preprocessing_DataAugmentation_Biclass(Mini_MIAS_CLAHE_Cropped_Images_Normal, Mini_MIAS_CLAHE_Cropped_Images_Tumor, Mini_MIAS_CLAHE_Cropped_Images_Biclass)
    #Dataframe_final_CLAHE = Testing_CNN_Models_Biclass(Model_CNN, 'CLAHE', Images, Labels)

    #Images, Labels = preprocessing_DataAugmentation_Biclass(Mini_MIAS_HE_Cropped_Images_Normal, Mini_MIAS_HE_Cropped_Images_Tumor, Mini_MIAS_HE_Cropped_Images_Biclass)
    #Dataframe_final_HE = Testing_CNN_Models_Biclass(Model_CNN, 'HE', Images, Labels)

    #Images, Labels = preprocessing_DataAugmentation_Biclass(Mini_MIAS_UM_Cropped_Images_Normal, Mini_MIAS_UM_Cropped_Images_Tumor, Mini_MIAS_UM_Cropped_Images_Biclass)
    #Dataframe_final_UM = Testing_CNN_Models_Biclass(Model_CNN, 'UM', Images, Labels)

    #Images, Labels = preprocessing_DataAugmentation_Biclass(Mini_MIAS_CS_Cropped_Images_Normal, Mini_MIAS_CS_Cropped_Images_Tumor, Mini_MIAS_CS_Cropped_Images_Biclass)
    #Dataframe_final_CS = Testing_CNN_Models_Biclass(Model_CNN, 'CS', Images, Labels)

    #concat_dataframe(Dataframe_final_NT, Dataframe_final_NO, Dataframe_final_CLAHE, Dataframe_final_HE, Dataframe_final_UM, Dataframe_final_CS, Folder = Biclass_Data_CSV, Class = 'Biclass', Technique = 'All techniques')

    #"""

    #"""

    #Images, Labels = preprocessing_DataAugmentation_Multiclass(Mini_MIAS_NT_Cropped_Images_Normal, Mini_MIAS_NT_Cropped_Images_Benign, Mini_MIAS_NT_Cropped_Images_Malignant, Mini_MIAS_NT_Cropped_Images_Multiclass)
    #Dataframe_final_NT = Testing_CNN_Models_Multiclass(Model_CNN_R, 'NT', Images, Labels)

    #Images, Labels = preprocessing_DataAugmentation_Multiclass(Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Benign, Mini_MIAS_NO_Cropped_Images_Malignant, Mini_MIAS_NO_Cropped_Images_Multiclass)
    #Dataframe_final_NO = Testing_CNN_Models_Multiclass(Model_CNN_R, 'NO', Images, Labels)

    #Images, Labels = preprocessing_DataAugmentation_Multiclass(Mini_MIAS_CLAHE_Cropped_Images_Normal, Mini_MIAS_CLAHE_Cropped_Images_Benign, Mini_MIAS_CLAHE_Cropped_Images_Malignant, Mini_MIAS_CLAHE_Cropped_Images_Multiclass)
    #Dataframe_final_CLAHE = Testing_CNN_Models_Multiclass(Model_CNN_R, 'CLAHE', Images, Labels)

    #Images, Labels = preprocessing_DataAugmentation_Multiclass(Mini_MIAS_HE_Cropped_Images_Normal, Mini_MIAS_HE_Cropped_Images_Benign, Mini_MIAS_HE_Cropped_Images_Malignant, Mini_MIAS_HE_Cropped_Images_Multiclass)
    #Dataframe_final_HE = Testing_CNN_Models_Multiclass(Model_CNN_R, 'HE', Images, Labels)

    #Images, Labels = preprocessing_DataAugmentation_Multiclass(Mini_MIAS_UM_Cropped_Images_Normal, Mini_MIAS_UM_Cropped_Images_Benign, Mini_MIAS_UM_Cropped_Images_Malignant, Mini_MIAS_UM_Cropped_Images_Multiclass)
    #Dataframe_final_UM = Testing_CNN_Models_Multiclass(Model_CNN_R, 'UM', Images, Labels)

    #Images, Labels = preprocessing_DataAugmentation_Multiclass(Mini_MIAS_CS_Cropped_Images_Normal, Mini_MIAS_CS_Cropped_Images_Benign, Mini_MIAS_CS_Cropped_Images_Malignant, Mini_MIAS_CS_Cropped_Images_Multiclass)
    #Dataframe_final_CS = Testing_CNN_Models_Multiclass(Model_CNN_R, 'CS', Images, Labels)

    #concat_dataframe(Dataframe_final_NT, Dataframe_final_NO, Dataframe_final_CLAHE, Dataframe_final_HE, Dataframe_final_UM, Dataframe_final_CS, Folder = Multiclass_Data_CSV, Class = 'Multiclass', Technique = 'All techniques')
    
    #"""

    #Split_Folders_Each_Technique()

    #training_testing_validation_from_directory(Mini_MIAS_NO_Cropped_Images_Biclass + '_Split')
    training_testing_validation_from_directory(Model_CNN_test, Mini_MIAS_NO_Cropped_Images_Multiclass + '_Split', 'NO')

if __name__ == "__main__":
    main()
