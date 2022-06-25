import os
import pandas as pd

from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Tumor
from Mini_MIAS_1_Folders import General_Data_CSV
from Mini_MIAS_1_Folders import General_Data_Model

from Mini_MIAS_6_Kmeans import featureExtraction
from Mini_MIAS_6_Kmeans import Kmeans

def preprocessing_Kmeans_GLCM_Tumor():

    # Parameters

    png_format = '.png' # TExtension 
    tumor_images_string = 'Tumor'   # Tumor label 
    tumor_images_label = 0   # Tumor label 
    features_extraction_technique = 'GLCM'   # Technique for features extraction

    TexturesFeatureGLCM = featureExtraction(folder = Mini_MIAS_NT_Cropped_Images_Tumor, label = tumor_images_label, extension = png_format)
    Data, X_Data, Filename = TexturesFeatureGLCM.TexturesFeatureGLCM()

    pd.set_option('display.max_rows', Data.shape[0] + 1)
    dst = 'GLCM' + '_Features_' + str(tumor_images_string) + '.csv'
    dstPath = os.path.join(General_Data_CSV, dst)
    Data.to_csv(dstPath)

    KMeans = Kmeans(folderCSV = General_Data_CSV, folderpic = General_Data_Model, name = TFI, X = X_Data, clusters = 2, filename = Filename, severity = Tumor)
    DataframeData = KMeans.KmeansFunction()

    KMeansDataRemoved = Kmeans(folder = Mini_MIAS_NT_Cropped_Images_Tumor, folderCSV = General_Data_CSV, name = TFI, df = DataframeData, CR = 1, severity = Tumor)
    KMeansDataRemoved.RemoveDataKmeans()
