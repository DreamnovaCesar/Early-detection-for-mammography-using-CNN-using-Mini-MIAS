import os
import pandas as pd

from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Benign
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Malignant
from Mini_MIAS_1_Folders import General_Data_CSV
from Mini_MIAS_1_Folders import General_Data_Model

from Mini_MIAS_7_Extract_Feature import featureExtraction
from Mini_MIAS_6_Kmeans import kmeans_function
from Mini_MIAS_6_Kmeans import kmeans_remove_data

def preprocessing_Kmeans_GLCM_Benign():

    # * General parameters

    PNG_format = '.png' 
    Benign_images_string = 'Benign'  
    Benign_images_label = 0  
    Features_extraction_technique = 'GLCM'  
    Clusters_kmeans = 2
    Cluster_to_remove = 1

    # * With this class we extract the features using GLCM
    Feature_extraction = featureExtraction(Folder = Mini_MIAS_NT_Cropped_Images_Benign, Label = Benign_images_label, Format = PNG_format)
    Benign_dataframe_GLCM, Benign_X_GLCM, All_filenames = Feature_extraction.textures_Feature_GLCM_from_folder()

    #pd.set_option('display.max_rows', benign_dataframe_GLCM.shape[0] + 1)
    Benign_dataframe_name = str(Features_extraction_technique) + '_Features_' + str(Benign_images_string) + '.csv'
    Benign_dataframe_path = os.path.join(General_Data_CSV, Benign_dataframe_name)
    Benign_dataframe_GLCM.to_csv(Benign_dataframe_path)

    # * We remove the cluster chosen, and a new dataframe is created
    Kmeans_dataframe = kmeans_function(General_Data_CSV, General_Data_Model, Features_extraction_technique, Benign_X_GLCM, Clusters_kmeans, All_filenames, Benign_images_string)
    Kmeans_dataframe_removed_data = kmeans_remove_data(Mini_MIAS_NT_Cropped_Images_Benign, General_Data_CSV, Features_extraction_technique, Kmeans_dataframe, Cluster_to_remove, Benign_images_string)


def preprocessing_Kmeans_GLCM_Malignant():

    # * General parameters

    PNG_format = '.png' # TExtension 
    Malignant_images_string = 'Malignant'   # Tumor label 
    Malignant_images_label = 1   # Malignant label 
    Features_extraction_technique = 'GLCM'   # Technique for features extraction string
    Clusters_kmeans = 2
    Cluster_to_remove = 1

    # * With this class we extract the features using GLCM
    Feature_extraction = featureExtraction(Folder = Mini_MIAS_NT_Cropped_Images_Malignant, Label = Malignant_images_label, Format = PNG_format)
    Malignant_dataframe_GLCM, Malignantn_X_GLCM, All_filenames = Feature_extraction.textures_Feature_GLCM_from_folder()

    #pd.set_option('display.max_rows', Data.shape[0] + 1)
    Malignant_dataframe_name = str(Features_extraction_technique) + '_Features_' + str(Malignant_images_string) + '.csv'
    Malignant_dataframe_path = os.path.join(General_Data_CSV, Malignant_dataframe_name)
    Malignant_dataframe_GLCM.to_csv(Malignant_dataframe_path)

    # * We remove the cluster chosen, and a new dataframe is created
    Kmeans_dataframe = kmeans_function(General_Data_CSV, General_Data_Model, Features_extraction_technique, Malignantn_X_GLCM, Clusters_kmeans, All_filenames, Malignant_images_string)
    Kmeans_dataframe_removed_data = kmeans_remove_data(Mini_MIAS_NT_Cropped_Images_Malignant, General_Data_CSV, Features_extraction_technique, Kmeans_dataframe, Cluster_to_remove, Malignant_images_string)




