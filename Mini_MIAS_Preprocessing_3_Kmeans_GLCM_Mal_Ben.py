import os
import pandas as pd

from Mini_MIAS_1_Folders import NTCropped_Images_Benign
from Mini_MIAS_1_Folders import NTCropped_Images_Malignant
from Mini_MIAS_1_Folders import General_Data_CSV
from Mini_MIAS_1_Folders import General_Data_Model

from Mini_MIAS_6_Kmeans import featureExtraction
from Mini_MIAS_6_Kmeans import Kmeans

def preprocessing_Kmeans_GLCM_Benign():

    # Parameters

    Extension_png = '.png' # TExtension 
    Benign = 'Benign'   # Tumor label 
    IB = 0   # Malignant label 
    TFI = 'GLCM'   # Technique for features extraction

    TexturesFeatureGLCM = featureExtraction(folder = NTCropped_Images_Benign, label = IB, extension = Extension_png)
    Data, X_Data, Filename = TexturesFeatureGLCM.TexturesFeatureGLCM()

    pd.set_option('display.max_rows', Data.shape[0] + 1)
    dst = TFI + '_Features_' + str(Benign) + '.csv'
    dstPath = os.path.join(General_Data_CSV, dst)
    Data.to_csv(dstPath)

    KMeans = Kmeans(folderCSV = General_Data_CSV, folderpic = General_Data_Model, name = TFI, X = X_Data, clusters = 2, filename = Filename, severity = Tumor)
    DataframeData = KMeans.KmeansFunction()

    KMeansDataRemoved = Kmeans(folder = NTCropped_Images_Benign, folderCSV = General_Data_CSV, name = TFI, df = DataframeData, CR = 1, severity = Tumor)
    KMeansDataRemoved.RemoveDataKmeans()

def preprocessing_Kmeans_GLCM_Malignant():

    # Parameters

    Extension_png = '.png' # TExtension 
    Malignant = 'Malignant'   # Tumor label 
    IM = 1   # Malignant label 
    TFI = 'GLCM'   # Technique for features extraction

    TexturesFeatureGLCM = featureExtraction(folder = NTCropped_Images_Malignant, label = IM, extension = Extension_png)
    Data, X_Data, Filename = TexturesFeatureGLCM.TexturesFeatureGLCM()

    pd.set_option('display.max_rows', Data.shape[0] + 1)
    dst = TFI + '_Features_' + str(Malignant) + '.csv'
    dstPath = os.path.join(General_Data_CSV, dst)
    Data.to_csv(dstPath)

    KMeans = Kmeans(folderCSV = General_Data_CSV, folderpic = General_Data_Model, name = TFI, X = X_Data, clusters = 2, filename = Filename, severity = Tumor)
    DataframeData = KMeans.KmeansFunction()

    KMeansDataRemoved = Kmeans(folder = NTCropped_Images_Malignant, folderCSV = General_Data_CSV, name = TFI, df = DataframeData, CR = 1, severity = Tumor)
    KMeansDataRemoved.RemoveDataKmeans()




