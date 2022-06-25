
########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from Mini_MIAS_2_General_Functions import dataframe_csv
from Mini_MIAS_3_Image_Processing import ImageProcessing

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from Mini_MIAS_1_Folders import Biclass_Data_CSV
from Mini_MIAS_1_Folders import Multiclass_Data_CSV

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from Mini_MIAS_1_Folders import NTCropped_Images_Normal
from Mini_MIAS_1_Folders import NTCropped_Images_Tumor
from Mini_MIAS_1_Folders import NTCropped_Images_Benign
from Mini_MIAS_1_Folders import NTCropped_Images_Malignant

from Mini_MIAS_1_Folders import NOCropped_Images_Normal
from Mini_MIAS_1_Folders import NOCropped_Images_Tumor
from Mini_MIAS_1_Folders import NOCropped_Images_Benign
from Mini_MIAS_1_Folders import NOCropped_Images_Malignant


def preprocessing_technique_Biclass(newtechnique):

    # Parameters for normalization 

    Normal = 'Normal' # Normal label
    Tumor = 'Tumor'   # Tumor label  

    IN = 0  # Normal class
    IT = 1  # Tumor class

    Biclass = 'Biclass' # Biclass label

    Normalization_Normal = ImageProcessing(folder = NTCropped_Images_Normal, newfolder = NOCropped_Images_Normal, severity = Normal, label = IN)
    Normalization_Tumor = ImageProcessing(folder = NTCropped_Images_Tumor, newfolder = NOCropped_Images_Tumor, severity = Tumor, label = IT)

    if newtechnique == 'NO':
        DataFrame_Normal = Normalization_Normal.Normalization()
        DataFrame_Tumor = Normalization_Tumor.Normalization()

    elif newtechnique == 'CLAHE':
        DataFrame_Normal = Normalization_Normal.CLAHE()
        DataFrame_Tumor = Normalization_Tumor.CLAHE()

    elif newtechnique == 'HE':
        DataFrame_Normal = Normalization_Normal.HistogramEqualization()
        DataFrame_Tumor = Normalization_Tumor.HistogramEqualization()

    elif newtechnique == 'UM':
        DataFrame_Normal = Normalization_Normal.UnsharpMasking()
        DataFrame_Tumor = Normalization_Tumor.UnsharpMasking()

    elif newtechnique == 'CS':
        DataFrame_Normal = Normalization_Normal.ContrastStretching()
        DataFrame_Tumor = Normalization_Tumor.ContrastStretching()
        
    else:
        print("technique chosen does not exist")

    DataframeSave(DataFrame_Normal, DataFrame_Tumor, folder = Biclass_Data_CSV, Class = Biclass, technique = newtechnique)

def preprocessing_technique_Multiclass(newtechnique):

    # Parameters for normalization

    Normal = 'Normal'   # Normal label 
    Benign = 'Benign'   # Benign label
    Malignant = 'Malignant' # Malignant label

    IN = 0  # Normal class
    IB = 1  # Benign class
    IM = 2  # Malignant class

    Multiclass = 'Multiclass' # Multiclass label

    Normalization_Normal = ImageProcessing(folder = NTCropped_Images_Normal, newfolder = NOCropped_Images_Normal, severity = Normal, label = IN)
    Normalization_Benign = ImageProcessing(folder = NTCropped_Images_Benign, newfolder = NOCropped_Images_Benign, severity = Benign, label = IB)
    Normalization_Malignant = ImageProcessing(folder = NTCropped_Images_Malignant, newfolder = NOCropped_Images_Malignant, severity = Malignant, label = IM)

    if newtechnique == 'NO':
        DataFrame_Normal = Normalization_Normal.Normalization()
        DataFrame_Benign = Normalization_Benign.Normalization()
        DataFrame_Malignant = Normalization_Malignant.Normalization()

    elif newtechnique == 'CLAHE':
        DataFrame_Normal = Normalization_Normal.CLAHE()
        DataFrame_Benign = Normalization_Benign.CLAHE()
        DataFrame_Malignant = Normalization_Malignant.CLAHE()

    elif newtechnique == 'HE':
        DataFrame_Normal = Normalization_Normal.HistogramEqualization()
        DataFrame_Benign = Normalization_Benign.HistogramEqualization()
        DataFrame_Malignant = Normalization_Malignant.HistogramEqualization()

    elif newtechnique == 'UM':
        DataFrame_Normal = Normalization_Normal.UnsharpMasking()
        DataFrame_Benign = Normalization_Benign.UnsharpMasking()
        DataFrame_Malignant = Normalization_Malignant.UnsharpMasking()

    elif newtechnique == 'CS':
        DataFrame_Normal = Normalization_Normal.ContrastStretching()
        DataFrame_Benign = Normalization_Benign.ContrastStretching()
        DataFrame_Malignant = Normalization_Malignant.ContrastStretching()

    else:
        raise ValueError("Technique does not exist")

    DataframeSave(DataFrame_Normal, DataFrame_Benign, DataFrame_Malignant, folder = Multiclass_Data_CSV, Class = Multiclass, technique = newtechnique)

preprocessing_technique_Biclass('NO')