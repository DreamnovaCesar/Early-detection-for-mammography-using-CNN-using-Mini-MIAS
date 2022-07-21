
########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from Mini_MIAS_2_General_Functions import concat_dataframe
from Mini_MIAS_3_Image_Processing import ImageProcessing

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from Mini_MIAS_1_Folders import Biclass_Data_CSV
from Mini_MIAS_1_Folders import Multiclass_Data_CSV

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

def preprocessing_technique_Biclass(New_technique, Folder_normal, Folder_tumor, New_folder_normal, New_folder_tumor):

    # * Parameters for normalization 

    # * Labels
    Label_Normal = 'Normal' 
    Label_Tumor = 'Tumor'   

    # * Classes
    Normal_images_class = 0 
    Tumor_images_class = 1 

    # * Problem class
    Biclass = 'Biclass' # Biclass label

    # * Image processing class
    Normalization_Normal = ImageProcessing(Folder = Folder_normal, Newfolder = New_folder_normal, Severity = Label_Normal, Label = Normal_images_class)
    Normalization_Tumor = ImageProcessing(Folder =  Folder_tumor, Newfolder = New_folder_tumor, Severity = Label_Tumor, Label = Tumor_images_class)

    # * Choose the technique utilized for the test
    if New_technique == 'NO':
        DataFrame_Normal = Normalization_Normal.normalize_technique()
        DataFrame_Tumor = Normalization_Tumor.normalize_technique()

    elif New_technique == 'CLAHE':
        DataFrame_Normal = Normalization_Normal.CLAHE_technique()
        DataFrame_Tumor = Normalization_Tumor.CLAHE_technique()

    elif New_technique == 'HE':
        DataFrame_Normal = Normalization_Normal.histogram_equalization_technique()
        DataFrame_Tumor = Normalization_Tumor.histogram_equalization_technique()

    elif New_technique == 'UM':
        DataFrame_Normal = Normalization_Normal.unsharp_masking_technique()
        DataFrame_Tumor = Normalization_Tumor.unsharp_masking_technique()

    elif New_technique == 'CS':
        DataFrame_Normal = Normalization_Normal.contrast_stretching_technique()
        DataFrame_Tumor = Normalization_Tumor.contrast_stretching_technique()
        
    else:
        raise ValueError("Choose a new technique")      #! Alert

    # * Concatenate dataframes with this function
    concat_dataframe(DataFrame_Normal, DataFrame_Tumor, Folder = Biclass_Data_CSV, Class = Biclass, Technique = New_technique)

def preprocessing_technique_Multiclass(New_technique, Folder_normal, Folder_benign, Folder_malignant, New_folder_normal, New_folder_benign, New_folder_malignant):

    # * Parameters for normalization

    # * Labels
    Label_Normal = 'Normal'   # Normal label 
    Label_Benign = 'Benign'   # Benign label
    Label_Malignant = 'Malignant' # Malignant label

    # * Classes
    Normal_images_class = 0 # Normal class
    Benign_images_class = 1 # Tumor class
    Malignant_images_class = 2 # Tumor class

    # * Problem class
    Multiclass = 'Multiclass' # Multiclass label

    Normalization_Normal = ImageProcessing(Folder = Folder_normal, Newfolder = New_folder_normal, Severity = Label_Normal, Label = Normal_images_class)
    Normalization_Benign = ImageProcessing(Folder = Folder_benign, Newfolder = New_folder_benign, Severity = Label_Benign, Label = Benign_images_class)
    Normalization_Malignant = ImageProcessing(Folder = Folder_malignant, Newfolder = New_folder_malignant, Severity = Label_Malignant, Label = Malignant_images_class)

    if New_technique == 'NO':
        DataFrame_Normal = Normalization_Normal.normalize_technique()
        DataFrame_Benign = Normalization_Benign.normalize_technique()
        DataFrame_Malignant = Normalization_Malignant.normalize_technique()

    elif New_technique == 'CLAHE':
        DataFrame_Normal = Normalization_Normal.CLAHE_technique()
        DataFrame_Benign = Normalization_Benign.CLAHE_technique()
        DataFrame_Malignant = Normalization_Malignant.CLAHE_technique()

    elif New_technique == 'HE':
        DataFrame_Normal = Normalization_Normal.histogram_equalization_technique()
        DataFrame_Benign = Normalization_Benign.histogram_equalization_technique()
        DataFrame_Malignant = Normalization_Malignant.histogram_equalization_technique()

    elif New_technique == 'UM':
        DataFrame_Normal = Normalization_Normal.unsharp_masking_technique()
        DataFrame_Benign = Normalization_Benign.unsharp_masking_technique()
        DataFrame_Malignant = Normalization_Malignant.unsharp_masking_technique()

    elif New_technique == 'CS':
        DataFrame_Normal = Normalization_Normal.contrast_stretching_technique()
        DataFrame_Benign = Normalization_Benign.contrast_stretching_technique()
        DataFrame_Malignant = Normalization_Malignant.contrast_stretching_technique()

    else:
        raise ValueError("Choose a new technique")    #! Alert

    # * Concatenate dataframes with this function
    concat_dataframe(DataFrame_Normal, DataFrame_Benign, DataFrame_Malignant, Folder = Multiclass_Data_CSV, Class = Multiclass, Technique = New_technique)
