import os
import pandas as pd

from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Tumor
from Mini_MIAS_1_Folders import General_Data_CSV
from Mini_MIAS_1_Folders import General_Data_Model

from Mini_MIAS_6_Kmeans import featureExtraction
from Mini_MIAS_6_Kmeans import kmeans_function
from Mini_MIAS_6_Kmeans import kmeans_remove_data

def preprocessing_Kmeans_GLCM_Tumor():

    # Parameters

    PNG_format = '.png' # TExtension 
    Tumor_images_string = 'Tumor'   # Benign label string
    Tumor_images_label = 0   # Benign string
    Features_extraction_technique = 'GLCM'   # Technique for features extraction string
    Clusters_kmeans = 2
    Cluster_to_remove = 1

    Feature_extraction = featureExtraction(folder = Mini_MIAS_NT_Cropped_Images_Tumor, label = Tumor_images_label, extension = PNG_format)
    Tumor_dataframe_GLCM, Tumor_X_GLCM, All_filenames = Feature_extraction.textures_Feature_GLCM_from_images()

    #pd.set_option('display.max_rows', benign_dataframe_GLCM.shape[0] + 1)
    Tumor_dataframe_name = str(Features_extraction_technique) + '_Features_' + str(Tumor_images_string) + '.csv'
    Tumor_dataframe_path = os.path.join(General_Data_CSV, Tumor_dataframe_name)
    Tumor_dataframe_GLCM.to_csv(Tumor_dataframe_path)

    Kmeans_dataframe = kmeans_function(General_Data_CSV, General_Data_Model, Features_extraction_technique, Tumor_X_GLCM, Clusters_kmeans, All_filenames, Tumor_images_string)
    Kmeans_dataframe_removed_data = kmeans_remove_data(Mini_MIAS_NT_Cropped_Images_Tumor, General_Data_CSV, Features_extraction_technique, Kmeans_dataframe, Cluster_to_remove, Tumor_images_string)
