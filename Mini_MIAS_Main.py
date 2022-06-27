from Mini_MIAS_Preprocessing_1_ChangeExtension import preprocessing_ChangeExtension

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

from Mini_MIAS_7_CNN_Architectures import ResNet50_PreTrained

from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Normal
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Tumor
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Benign
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Malignant

from Mini_MIAS_1_Folders import Mini_MIAS_NO_Cropped_Images_Normal
from Mini_MIAS_1_Folders import Mini_MIAS_NO_Cropped_Images_Tumor
from Mini_MIAS_1_Folders import Mini_MIAS_NO_Cropped_Images_Benign
from Mini_MIAS_1_Folders import Mini_MIAS_NO_Cropped_Images_Malignant

def main():

    Model_Tested = ResNet50_PreTrained

    #preprocessing_ChangeExtension()
    #preprocessing_Cropped_MIAS_Mammograms()
    #preprocessing_Kmeans_GLCM_Tumor()
    #preprocessing_Kmeans_GLCM_Benign()
    #preprocessing_Kmeans_GLCM_Malignant()
    #preprocessing_Resize()
    #preprocessing_technique_Biclass('NO', Mini_MIAS_NT_Cropped_Images_Normal, Mini_MIAS_NT_Cropped_Images_Tumor, Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Tumor)
    #preprocessing_technique_Multiclass( 'NO', Mini_MIAS_NT_Cropped_Images_Normal, Mini_MIAS_NT_Cropped_Images_Benign, Mini_MIAS_NT_Cropped_Images_Malignant, 
    #                                        Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Benign, Mini_MIAS_NO_Cropped_Images_Malignant)

    #preprocessing_DataAugmentation_Biclass()
    Images, Labels = preprocessing_DataAugmentation_Multiclass(Mini_MIAS_NO_Cropped_Images_Normal, Mini_MIAS_NO_Cropped_Images_Benign, Mini_MIAS_NO_Cropped_Images_Malignant)

    #Testing_CNN_Models_Biclass()
    Testing_CNN_Models_Multiclass(Model_Tested, Images, Labels)
    
if __name__ == "__main__":
    main()
