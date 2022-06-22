
########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from Mini_MIAS_1_Folders import NTCropped_Images_Normal
from Mini_MIAS_1_Folders import NTCropped_Images_Tumor

from Mini_MIAS_1_Folders import NOCropped_Images_Normal
from Mini_MIAS_1_Folders import NOCropped_Images_Tumor

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from Mini_MIAS_4_Data_Augmentation import dataAugmentation

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

def preprocessing_DataAugmentation_Biclass():

    # Parameters

    Images = []
    Labels = []

    NNormal = 20 # The number of normal images
    NTumor = 40  # The number of tumor images

    Normal = 'Normal'  # Normal label
    Tumor = 'Tumor'    # Tumor label

    IN = 0 # Normal class
    IT = 1 # Tumor class

    DataAug_0 = dataAugmentation(folder = NTCropped_Images_Normal, severity = Normal, sampling = NNormal, label = IN, nfsave = False)
    DataAug_1 = dataAugmentation(folder = NTCropped_Images_Tumor, severity = Tumor, sampling = NTumor, label = IT, nfsave = False)

    DataAug_2 = dataAugmentation(folder = NOCropped_Images_Normal, severity = Normal, sampling = NNormal, label = IN, nfsave = False)
    DataAug_3 = dataAugmentation(folder = NOCropped_Images_Tumor, severity = Tumor, sampling = NTumor, label = IT, nfsave = False)

    Images_Normal, Labels_Normal = DataAug_0.DataAugmentation()
    Images_Tumor, Labels_Tumor = DataAug_1.DataAugmentation()

    NOImages_Normal, NOLabels_Normal = DataAug_2.DataAugmentation()
    NOImages_Tumor, NOLabels_Tumor = DataAug_3.DataAugmentation()

    Images.append(Images_Normal)
    Images.append(Images_Tumor)

    Images.append(NOImages_Normal)
    Images.append(NOImages_Tumor)

    Labels.append(Labels_Normal)
    Labels.append(Labels_Tumor)

    Labels.append(NOLabels_Normal)
    Labels.append(NOLabels_Tumor)

    return Images, Labels

preprocessing_DataAugmentation_Biclass()