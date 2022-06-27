
########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from Mini_MIAS_2_General_Functions import concat_dataframe
from Mini_MIAS_3_Image_Processing import ImageProcessing

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from Mini_MIAS_1_Folders import Biclass_Data_CSV
from Mini_MIAS_1_Folders import Multiclass_Data_CSV

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

def preprocessing_technique_Biclass(New_technique, Folder_normal, Folder_tumor, New_folder_normal, New_folder_tumor):

    # Parameters for normalization 

    Label_Normal = 'Normal' # Normal label
    Label_Tumor = 'Tumor'   # Tumor label  

    Normal_images_label = 0 # Normal class
    Tumor_images_label = 1 # Tumor class

    Biclass = 'Biclass' # Biclass label

    Normalization_Normal = ImageProcessing(folder = Folder_normal, newfolder = New_folder_normal, severity = Label_Normal, label = Normal_images_label)
    Normalization_Tumor = ImageProcessing(folder =  Folder_tumor, newfolder = New_folder_tumor, severity = Label_Tumor, label = Tumor_images_label)

    if New_technique == 'NO':
        DataFrame_Normal = Normalization_Normal.normalize_technique()
        DataFrame_Tumor = Normalization_Tumor.normalize_technique()

    elif New_technique == 'CLAHE':
        DataFrame_Normal = Normalization_Normal.CLAHE_technique()
        DataFrame_Tumor = Normalization_Tumor.CLAHE_technique()

    elif New_technique == 'HE':
        DataFrame_Normal = Normalization_Normal.HistogramEqualization()
        DataFrame_Tumor = Normalization_Tumor.HistogramEqualization()

    elif New_technique == 'UM':
        DataFrame_Normal = Normalization_Normal.UnsharpMasking()
        DataFrame_Tumor = Normalization_Tumor.UnsharpMasking()

    elif New_technique == 'CS':
        DataFrame_Normal = Normalization_Normal.ContrastStretching()
        DataFrame_Tumor = Normalization_Tumor.ContrastStretching()
        
    else:
        raise ValueError("Technique does not exist")

    concat_dataframe(DataFrame_Normal, DataFrame_Tumor, folder = Biclass_Data_CSV, Class = Biclass, technique = New_technique)

def preprocessing_technique_Multiclass(New_technique, Folder_normal, Folder_benign, Folder_malignant, New_folder_normal, New_folder_benign, New_folder_malignant):

    # Parameters for normalization

    Label_Normal = 'Normal'   # Normal label 
    Label_Benign = 'Benign'   # Benign label
    Label_Malignant = 'Malignant' # Malignant label

    Normal_images_label = 0 # Normal class
    Benign_images_label = 1 # Tumor class
    Malignant_images_label = 2 # Tumor class

    Multiclass = 'Multiclass' # Multiclass label

    Normalization_Normal = ImageProcessing(folder = Folder_normal, newfolder = New_folder_normal, severity = Label_Normal, label = Normal_images_label)
    Normalization_Benign = ImageProcessing(folder = Folder_benign, newfolder = New_folder_benign, severity = Label_Benign, label = Benign_images_label)
    Normalization_Malignant = ImageProcessing(folder = Folder_malignant, newfolder = New_folder_malignant, severity = Label_Malignant, label = Malignant_images_label)

    if New_technique == 'NO':
        DataFrame_Normal = Normalization_Normal.normalize_technique()
        DataFrame_Benign = Normalization_Benign.normalize_technique()
        DataFrame_Malignant = Normalization_Malignant.normalize_technique()

    elif New_technique == 'CLAHE':
        DataFrame_Normal = Normalization_Normal.CLAHE_technique()
        DataFrame_Benign = Normalization_Benign.CLAHE_technique()
        DataFrame_Malignant = Normalization_Malignant.CLAHE_technique()

    elif New_technique == 'HE':
        DataFrame_Normal = Normalization_Normal.HistogramEqualization()
        DataFrame_Benign = Normalization_Benign.HistogramEqualization()
        DataFrame_Malignant = Normalization_Malignant.HistogramEqualization()

    elif New_technique == 'UM':
        DataFrame_Normal = Normalization_Normal.UnsharpMasking()
        DataFrame_Benign = Normalization_Benign.UnsharpMasking()
        DataFrame_Malignant = Normalization_Malignant.UnsharpMasking()

    elif New_technique == 'CS':
        DataFrame_Normal = Normalization_Normal.ContrastStretching()
        DataFrame_Benign = Normalization_Benign.ContrastStretching()
        DataFrame_Malignant = Normalization_Malignant.ContrastStretching()

    else:
        raise ValueError("Technique does not exist")

    concat_dataframe(DataFrame_Normal, DataFrame_Benign, DataFrame_Malignant, folder = Multiclass_Data_CSV, Class = Multiclass, technique = New_technique)
