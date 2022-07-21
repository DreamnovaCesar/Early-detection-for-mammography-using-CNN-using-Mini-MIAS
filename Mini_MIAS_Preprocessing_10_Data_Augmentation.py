
import numpy as np

from Mini_MIAS_4_Data_Augmentation import dataAugmentation

from Mini_MIAS_1_Folders import Mini_MIAS_NO_Cropped_Images_Biclass
from Mini_MIAS_1_Folders import Mini_MIAS_NO_Cropped_Images_Multiclass 
from Mini_MIAS_1_Folders import Mini_MIAS_NO_Cropped_Images_Normal
from Mini_MIAS_1_Folders import Mini_MIAS_NO_Cropped_Images_Tumor
from Mini_MIAS_1_Folders import Mini_MIAS_NO_Cropped_Images_Benign
from Mini_MIAS_1_Folders import Mini_MIAS_NO_Cropped_Images_Malignant

def preprocessing_DataAugmentation_Biclass(Folder_normal, Folder_tumor, Folder_destination):

    # * List to add images and labels.
    Images = []
    Labels = []

    # * General parameters
    #Iter_normal = 20 
    #Iter_tumor = 40 

    Iter_normal = 18 
    Iter_tumor = 34

    #Iter_normal = 2 
    #Iter_tumor = 4  

    Label_normal = 'Normal' 
    Label_tumor = 'Tumor'  

    Normal_images_class = 0 
    Tumor_images_class = 1 

    # * With this class we use the technique called data augmentation to create new images with their transformations
    Data_augmentation_normal = dataAugmentation(Folder = Folder_normal, NewFolder = Folder_destination, Severity = Label_normal, Sampling = Iter_normal, Label = Normal_images_class, Saveimages = True)
    Data_augmentation_tumor = dataAugmentation(Folder = Folder_tumor, NewFolder = Folder_destination, Severity = Label_tumor, Sampling = Iter_tumor, Label = Tumor_images_class, Saveimages = True)

    Images_Normal, Labels_Normal = Data_augmentation_normal.data_augmentation()
    Images_Tumor, Labels_Tumor = Data_augmentation_tumor.data_augmentation()

    # * Add the value in the lists already created

    Images_total = Images_Normal + Images_Tumor
    Labels_total = np.concatenate((Labels_Normal, Labels_Tumor), axis = None)

    print(len(Images_Normal))
    print(len(Images_Tumor))

    return Images_total, Labels_total
