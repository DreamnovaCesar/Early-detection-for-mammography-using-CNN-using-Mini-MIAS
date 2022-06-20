
########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from Mini_MIAS_2_General_Functions import DataframeSave
from Mini_MIAS_3_Image_Processing import ImageProcessing

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from Mini_MIAS_1_Folders import Biclass_Data_CSV
from Mini_MIAS_1_Folders import Multiclass_Data_CSV

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from Mini_MIAS_1_Folders import Cropped_Images_Normal
from Mini_MIAS_1_Folders import Cropped_Images_Tumor
from Mini_MIAS_1_Folders import Cropped_Images_Benign
from Mini_MIAS_1_Folders import Cropped_Images_Malignant

from Mini_MIAS_1_Folders import NOCropped_Images_Normal
from Mini_MIAS_1_Folders import NOCropped_Images_Tumor
from Mini_MIAS_1_Folders import NOCropped_Images_Benign
from Mini_MIAS_1_Folders import NOCropped_Images_Malignant


def preprocessing_Normalize_Biclass():

    # Parameters for normalization 

    Normal = 'Normal' # Normal label
    Tumor = 'Tumor'   # Tumor label  

    IN = 0  # Normal class
    IT = 1  # Tumor class

    Biclass = 'Biclass' # Biclass label
    Normalization = 'Normalization' # Multiclass label

    Normalization_Normal = ImageProcessing(folder = Cropped_Images_Normal, newfolder = NOCropped_Images_Normal, severity = Normal, label = IN)
    Normalization_Tumor = ImageProcessing(folder = Cropped_Images_Tumor, newfolder = NOCropped_Images_Tumor, severity = Tumor, label = IT)

    DataFrame_Normal = Normalization_Normal.Normalization()
    DataFrame_Tumor = Normalization_Tumor.Normalization()

    DataframeSave(DataFrame_Normal, DataFrame_Tumor, folder = Biclass_Data_CSV, Class = Biclass, technique = Normalization)

def preprocessing_Normalize_Multiclass():

    # Parameters for normalization

    Normal = 'Normal'   # Normal label 
    Benign = 'Benign'   # Benign label
    Malignant = 'Malignant' # Malignant label

    IN = 0  # Normal class
    IB = 1  # Benign class
    IM = 2  # Malignant class

    Multiclass = 'Multiclass' # Multiclass label
    Normalization = 'Normalization' # Multiclass label

    Normalization_Normal = ImageProcessing(folder = Cropped_Images_Normal, newfolder = NOCropped_Images_Normal, severity = Normal, label = IN)
    Normalization_Benign = ImageProcessing(folder = Cropped_Images_Benign, newfolder = NOCropped_Images_Benign, severity = Benign, label = IB)
    Normalization_Malignant = ImageProcessing(folder = Cropped_Images_Malignant, newfolder = NOCropped_Images_Malignant, severity = Malignant, label = IM)

    DataFrame_Normal = Normalization_Normal.Normalization()
    DataFrame_Benign = Normalization_Benign.Normalization()
    DataFrame_Malignant = Normalization_Malignant.Normalization()

    DataframeSave(DataFrame_Normal, DataFrame_Benign, DataFrame_Malignant, folder = Multiclass_Data_CSV, Class = Multiclass, technique = Normalization)
