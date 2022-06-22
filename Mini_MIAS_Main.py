from Mini_MIAS_Preprocessing_1_ChangeExtension import preprocessing_ChangeExtension

from Mini_MIAS_Preprocessing_2_Cropped_MIAS_Mammograms import preprocessing_Cropped_MIAS_Mammograms

from Mini_MIAS_Preprocessing_3_Kmeans_GLCM import preprocessing_Kmeans_GLCM_Tumor
from Mini_MIAS_Preprocessing_3_Kmeans_GLCM_Mal_Ben import preprocessing_Kmeans_GLCM_Benign
from Mini_MIAS_Preprocessing_3_Kmeans_GLCM_Mal_Ben import preprocessing_Kmeans_GLCM_Malignant

from Mini_MIAS_Preprocessing_4_Resize import preprocessing_Resize

from Mini_MIAS_Preprocessing_5_Select_technique import preprocessing_select_technique

from Mini_MIAS_Preprocessing_10_Data_Augmentation import preprocessing_DataAugmentation_Biclass
from Mini_MIAS_Preprocessing_10_Multi_Data_Augmentation import preprocessing_DataAugmentation_Multiclass

from Mini_MIAS_Preprocessing_11_CNN_Models import Testing_CNN_Models_Biclass
from Mini_MIAS_Preprocessing_11_Multi_CNN_Models import Testing_CNN_Models_Multiclass

def preprocessing():

    preprocessing_ChangeExtension()
    preprocessing_Cropped_MIAS_Mammograms()
    preprocessing_Kmeans_GLCM_Tumor()
    preprocessing_Kmeans_GLCM_Benign()
    preprocessing_Kmeans_GLCM_Malignant()
    preprocessing_Resize()
    preprocessing_select_technique()
    preprocessing_DataAugmentation_Biclass()
    preprocessing_DataAugmentation_Multiclass()

def testing():
    Testing_CNN_Models_Biclass()
    Testing_CNN_Models_Multiclass()

def main():

    preprocessing()
    testing()
    
if __name__ == "__main__":
    main()
