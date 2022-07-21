import numpy as np

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

from Mini_MIAS_Preprocessing_12_Data_Augmentation_Folder import split_Folders_Each_Technique

from Mini_MIAS_8_CNN_Architectures import MobileNet_Pretrained
from Mini_MIAS_8_CNN_Architectures import MobileNetV3Small_Pretrained
from Mini_MIAS_8_CNN_Architectures import MobileNetV3Large_Pretrained

from Mini_MIAS_8_CNN_Architectures import ResNet152_PreTrained
from Mini_MIAS_8_CNN_Architectures import ResNet152V2_PreTrained
from Mini_MIAS_8_CNN_Architectures import ResNet50_PreTrained

from Mini_MIAS_ML_FeaturesExtraction import Testing_ML_Models_Biclass_FOF
from Mini_MIAS_ML_FeaturesExtraction import Testing_ML_Models_Multiclass_FOF

from Mini_MIAS_ML_Functions import SVM
from Mini_MIAS_ML_Functions import Multi_SVM
from Mini_MIAS_ML_Functions import MLP
from Mini_MIAS_ML_Functions import KNN
from Mini_MIAS_ML_Functions import RF
from Mini_MIAS_ML_Functions import DT
from Mini_MIAS_ML_Functions import GBC

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

from sklearn.model_selection import train_test_split

def main():

    PGM = ".pgm"
    PNG = ".png"
    TIFF = ".tiff"

    Model_CNN = (MobileNet_Pretrained, MobileNetV3Small_Pretrained, MobileNetV3Large_Pretrained)
    Model_ML = (Multi_SVM, MLP, KNN, RF, DT, GBC)

    #preprocessing_ChangeFormat(PGM, PNG)
    #preprocessing_ChangeFormat(PGM, TIFF)
    #preprocessing_Cropped_MIAS_Mammograms()
    #preprocessing_Kmeans_GLCM_Tumor()
    #preprocessing_Kmeans_GLCM_Benign()
    #preprocessing_Kmeans_GLCM_Malignant()

    #preprocessing_Resize(Mini_MIAS_NT_Cropped_Images_Normal, Mini_MIAS_NT_Cropped_Images_Tumor, Mini_MIAS_NT_Cropped_Images_Benign, Mini_MIAS_NO_Cropped_Images_Malignant)
    #preprocessing_Resize(Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Tumor, Mini_MIAS_NO_Cropped_Images_Benign, Mini_MIAS_NO_Cropped_Images_Malignant)
    #preprocessing_Resize(Mini_MIAS_CLAHE_Cropped_Images_Normal, Mini_MIAS_CLAHE_Cropped_Images_Tumor, Mini_MIAS_CLAHE_Cropped_Images_Benign, Mini_MIAS_CLAHE_Cropped_Images_Malignant)
    #preprocessing_Resize(Mini_MIAS_HE_Cropped_Images_Normal, Mini_MIAS_HE_Cropped_Images_Tumor, Mini_MIAS_HE_Cropped_Images_Benign, Mini_MIAS_HE_Cropped_Images_Malignant)
    #preprocessing_Resize(Mini_MIAS_UM_Cropped_Images_Normal, Mini_MIAS_UM_Cropped_Images_Tumor, Mini_MIAS_UM_Cropped_Images_Benign, Mini_MIAS_UM_Cropped_Images_Malignant)
    #preprocessing_Resize(Mini_MIAS_CS_Cropped_Images_Normal, Mini_MIAS_CS_Cropped_Images_Tumor, Mini_MIAS_CS_Cropped_Images_Benign, Mini_MIAS_CS_Cropped_Images_Malignant)

    #preprocessing_technique_Biclass('NO', Mini_MIAS_NT_Cropped_Images_Normal, Mini_MIAS_NT_Cropped_Images_Tumor, Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Tumor)
    #preprocessing_technique_Biclass('CLAHE', Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Tumor, Mini_MIAS_CLAHE_Cropped_Images_Normal, Mini_MIAS_CLAHE_Cropped_Images_Tumor)
    #preprocessing_technique_Biclass('HE', Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Tumor, Mini_MIAS_HE_Cropped_Images_Normal, Mini_MIAS_HE_Cropped_Images_Tumor)
    #preprocessing_technique_Biclass('UM', Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Tumor, Mini_MIAS_UM_Cropped_Images_Normal, Mini_MIAS_UM_Cropped_Images_Tumor)
    #preprocessing_technique_Biclass('CS', Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Tumor, Mini_MIAS_CS_Cropped_Images_Normal, Mini_MIAS_CS_Cropped_Images_Tumor)
   

    #preprocessing_technique_Multiclass( 'NO', Mini_MIAS_NT_Cropped_Images_Normal, Mini_MIAS_NT_Cropped_Images_Benign, Mini_MIAS_NT_Cropped_Images_Malignant, 
    #                                        Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Benign, Mini_MIAS_NO_Cropped_Images_Malignant)
    #preprocessing_technique_Multiclass( 'CLAHE', Mini_MIAS_NT_Cropped_Images_Normal, Mini_MIAS_NT_Cropped_Images_Benign, Mini_MIAS_NT_Cropped_Images_Malignant, 
    #                                        Mini_MIAS_CLAHE_Cropped_Images_Normal, Mini_MIAS_CLAHE_Cropped_Images_Benign, Mini_MIAS_CLAHE_Cropped_Images_Malignant)
    #preprocessing_technique_Multiclass( 'HE', Mini_MIAS_NT_Cropped_Images_Normal, Mini_MIAS_NT_Cropped_Images_Benign, Mini_MIAS_NT_Cropped_Images_Malignant, 
    #                                        Mini_MIAS_HE_Cropped_Images_Normal, Mini_MIAS_HE_Cropped_Images_Benign, Mini_MIAS_HE_Cropped_Images_Malignant)
    #preprocessing_technique_Multiclass( 'UM', Mini_MIAS_NT_Cropped_Images_Normal, Mini_MIAS_NT_Cropped_Images_Benign, Mini_MIAS_NT_Cropped_Images_Malignant, 
    #                                        Mini_MIAS_UM_Cropped_Images_Normal, Mini_MIAS_UM_Cropped_Images_Benign, Mini_MIAS_UM_Cropped_Images_Malignant)
    #preprocessing_technique_Multiclass( 'CS', Mini_MIAS_NT_Cropped_Images_Normal, Mini_MIAS_NT_Cropped_Images_Benign, Mini_MIAS_NT_Cropped_Images_Malignant, 
    #                                        Mini_MIAS_CS_Cropped_Images_Normal, Mini_MIAS_CS_Cropped_Images_Benign, Mini_MIAS_CS_Cropped_Images_Malignant)

    Images, Labels = preprocessing_DataAugmentation_Biclass(Mini_MIAS_NT_Cropped_Images_Normal, Mini_MIAS_NT_Cropped_Images_Tumor, Mini_MIAS_NT_Cropped_Images_Biclass)
    Testing_CNN_Models_Biclass(Model_CNN, 'NT', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Biclass(Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Tumor, Mini_MIAS_NO_Cropped_Images_Biclass)
    Testing_CNN_Models_Biclass(Model_CNN, 'NO', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Biclass(Mini_MIAS_CLAHE_Cropped_Images_Normal, Mini_MIAS_CLAHE_Cropped_Images_Tumor, Mini_MIAS_CLAHE_Cropped_Images_Biclass)
    Testing_CNN_Models_Biclass(Model_CNN, 'CLAHE', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Biclass(Mini_MIAS_CLAHE_Cropped_Images_Normal, Mini_MIAS_CLAHE_Cropped_Images_Tumor, Mini_MIAS_CLAHE_Cropped_Images_Biclass)
    Testing_CNN_Models_Biclass(Model_CNN, 'HE', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Biclass(Mini_MIAS_CLAHE_Cropped_Images_Normal, Mini_MIAS_CLAHE_Cropped_Images_Tumor, Mini_MIAS_CLAHE_Cropped_Images_Biclass)
    Testing_CNN_Models_Biclass(Model_CNN, 'CLAHE', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Biclass(Mini_MIAS_CLAHE_Cropped_Images_Normal, Mini_MIAS_CLAHE_Cropped_Images_Tumor, Mini_MIAS_CLAHE_Cropped_Images_Biclass)
    Testing_CNN_Models_Biclass(Model_CNN, 'CLAHE', Images, Labels)




    Testing_CNN_Models_Multiclass(Model_CNN, 'NO', Images, Labels)



    Images, Labels = preprocessing_DataAugmentation_Biclass(Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Tumor, Mini_MIAS_NO_Cropped_Images_Biclass)


    #Testing_ML_Models_Biclass_FOF(Model_CNN, 'NO', Images, Labels)
    Testing_CNN_Models_Biclass(Model_CNN, 'NO', Images, Labels)

    #Images, Labels = preprocessing_DataAugmentation_Multiclass(Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Benign, Mini_MIAS_NO_Cropped_Images_Malignant)
    #Testing_ML_Models_Multiclass_FOF(Models, 'NO', Images, Labels)

    #Testing_CNN_Models_Biclass()
    #Testing_CNN_Models_Multiclass(Model_Tested, Images, Labels)
    
if __name__ == "__main__":
    main()
