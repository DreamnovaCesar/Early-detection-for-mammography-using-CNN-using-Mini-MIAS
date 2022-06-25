import os
import pandas as pd

from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Benign
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Malignant
from Mini_MIAS_1_Folders import General_Data_CSV
from Mini_MIAS_1_Folders import General_Data_Model

from Mini_MIAS_6_Kmeans import featureExtraction
from Mini_MIAS_6_Kmeans import Kmeans

def preprocessing_Kmeans_GLCM_Benign():

    # Parameters

    png_format = '.png' # TExtension 
    benign_images_string = 'Benign'   # Benign label string
    benign_images_label = 0   # Benign string
    features_extraction_technique = 'GLCM'   # Technique for features extraction string

    feature_extraction = featureExtraction(folder = Mini_MIAS_NT_Cropped_Images_Benign, label = benign_images_label, extension = png_format)
    benign_dataframe_GLCM, benign_X_GLCM, all_filenames = feature_extraction.textures_Feature_GLCM_from_images()

    #pd.set_option('display.max_rows', benign_dataframe_GLCM.shape[0] + 1)
    benign_dataframe_name = str(features_extraction_technique) + '_Features_' + str(benign_images_string) + '.csv'
    benign_dataframe_path = os.path.join(General_Data_CSV, benign_dataframe_name)
    benign_dataframe_GLCM.to_csv(benign_dataframe_path)

    Kmeans = Kmeans(folderCSV = General_Data_CSV, folderpic = General_Data_Model, name = features_extraction_technique, X = benign_X_GLCM, clusters = 2, filename = all_filenames, severity = benign_images_string)
    Kmeans_dataframe = Kmeans.Kmeans_function()

    KMeansDataRemoved = Kmeans(folder = Mini_MIAS_NT_Cropped_Images_Benign, folderCSV = General_Data_CSV, name = features_extraction_technique, df = Kmeans_dataframe, CR = 1, severity = benign_images_string)
    KMeansDataRemoved.Kmeans_remove_data()

def preprocessing_Kmeans_GLCM_Malignant():

    # Parameters

    Extension_png = '.png' # TExtension 
    malignant_images_string = 'Malignant'   # Tumor label 
    malignant_images_label = 1   # Malignant label 
    features_extraction_technique = 'GLCM'   # Technique for features extraction string

    feature_extraction = featureExtraction(folder = Mini_MIAS_NT_Cropped_Images_Malignant, label = malignant_images_label, extension = Extension_png)
    Data, X_Data, Filename = feature_extraction.textures_Feature_GLCM_from_images()

    pd.set_option('display.max_rows', Data.shape[0] + 1)
    dst = str(features_extraction_technique) + '_Features_' + str(malignant_images_string) + '.csv'
    dstPath = os.path.join(General_Data_CSV, dst)
    Data.to_csv(dstPath)

    KMeans = Kmeans(folderCSV = General_Data_CSV, folderpic = General_Data_Model, name = TFI, X = X_Data, clusters = 2, filename = Filename, severity = Tumor)
    DataframeData = KMeans.KmeansFunction()

    KMeansDataRemoved = Kmeans(folder = Mini_MIAS_NT_Cropped_Images_Malignant, folderCSV = General_Data_CSV, name = TFI, df = DataframeData, CR = 1, severity = Tumor)
    KMeansDataRemoved.RemoveDataKmeans()




