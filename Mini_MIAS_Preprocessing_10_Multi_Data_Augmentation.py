
########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from Mini_MIAS_1_Folders import NTCropped_Images_Normal
from Mini_MIAS_1_Folders import NTCropped_Images_Tumor
from Mini_MIAS_1_Folders import NTCropped_Images_Benign
from Mini_MIAS_1_Folders import NTCropped_Images_Malignant

from Mini_MIAS_1_Folders import NOCropped_Images_Normal
from Mini_MIAS_1_Folders import NOCropped_Images_Tumor
from Mini_MIAS_1_Folders import NOCropped_Images_Benign
from Mini_MIAS_1_Folders import NOCropped_Images_Malignant

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from Mini_MIAS_4_Data_Augmentation import dataAugmentation

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

def preprocessing_DataAugmentation_Multiclass():

    # Parameters

    Images = []
    Labels = []

    NNormal = 14 # The number of normal images
    NBenign = 55 # The number of benign images
    NMalignant = 70  # The number of malignant images

    Normal = 'Normal'  # Normal label
    Benign = 'Benign'  # Benign label
    Malignant = 'Malignant'    # Malignant label

    IN = 0  # Normal class
    IB = 1  # Normal class
    IT = 2  # Normal class

    DataAug_0 = dataAugmentation(folder = NTCropped_Images_Normal, severity = Normal, sampling = NNormal, label = IN, nfsave = False)
    DataAug_1 = dataAugmentation(folder = NTCropped_Images_Benign, severity = Benign, sampling = NBenign, label = IT, nfsave = False)
    DataAug_2 = dataAugmentation(folder = NTCropped_Images_Malignant, severity = Malignant, sampling = NMalignant, label = IT, nfsave = False)

    DataAug_3 = dataAugmentation(folder = NOCropped_Images_Normal, severity = Normal, sampling = NNormal, label = IN, nfsave = False)
    DataAug_4 = dataAugmentation(folder = NOCropped_Images_Benign, severity = Benign, sampling = NBenign, label = IT, nfsave = False)
    DataAug_5 = dataAugmentation(folder = NOCropped_Images_Malignant, severity = Malignant, sampling = NMalignant, label = IT, nfsave = False)

    Images_Normal, Labels_Normal = DataAug_0.DataAugmentation()
    Images_Benign, Labels_Benign = DataAug_1.DataAugmentation()
    Images_Malignant, Labels_Malignant = DataAug_2.DataAugmentation()

    NOImages_Normal, NOLabels_Normal = DataAug_3.DataAugmentation()
    NOImages_Benign, NOLabels_Benign = DataAug_4.DataAugmentation()
    NOImages_Malignant, NOLabels_Malignant = DataAug_5.DataAugmentation()

    Images.append(Images_Normal)
    Images.append(Images_Benign)
    Images.append(Images_Malignant)

    Images.append(NOImages_Normal)
    Images.append(NOImages_Benign)
    Images.append(NOImages_Malignant)

    Labels.append(Labels_Normal)
    Labels.append(Labels_Benign)
    Labels.append(Labels_Malignant)

    Labels.append(NOLabels_Normal)
    Labels.append(NOLabels_Benign)
    Labels.append(NOLabels_Malignant)

    return Images, Labels