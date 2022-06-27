
from Mini_MIAS_4_Data_Augmentation import dataAugmentation

def preprocessing_DataAugmentation_Multiclass(Folder_normal, Folder_benign, Folder_malignant):

    # Parameters

    Images = []
    Labels = []

    NNormal = 2 # The number of normal images
    NBenign = 70 # The number of benign images
    NMalignant = 90  # The number of malignant images

    Normal = 'Normal'  # Normal label
    Benign = 'Benign'  # Benign label
    Malignant = 'Malignant'    # Malignant label

    Normal_images_label = 0 # Normal class
    Benign_images_label = 1 # Tumor class
    Malignant_images_label = 2 # Tumor class

    Data_augmentation_normal = dataAugmentation(folder = Folder_normal, severity = Normal, sampling = NNormal, label = Normal_images_label, nfsave = False)
    Data_augmentation_benign = dataAugmentation(folder = Folder_benign, severity = Benign, sampling = NBenign, label = Benign_images_label, nfsave = False)
    Data_augmentation_malignant = dataAugmentation(folder = Folder_malignant, severity = Malignant, sampling = NMalignant, label = Malignant_images_label, nfsave = False)

    Images_Normal, Labels_Normal = Data_augmentation_normal.DataAugmentation()
    Images_Benign, Labels_Benign = Data_augmentation_benign.DataAugmentation()
    Images_Malignant, Labels_Malignant = Data_augmentation_malignant.DataAugmentation()

    Images.append(Images_Normal)
    Images.append(Images_Benign)
    Images.append(Images_Malignant)

    Labels.append(Labels_Normal)
    Labels.append(Labels_Benign)
    Labels.append(Labels_Malignant)

    return Images, Labels