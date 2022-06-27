
from Mini_MIAS_4_Data_Augmentation import dataAugmentation

def preprocessing_DataAugmentation_Biclass(Folder_normal, Folder_tumor):

    # Parameters

    Images = []
    Labels = []

    Iter_normal = 20 # The number of normal images
    Iter_tumor = 40  # The number of tumor images

    Label_normal = 'Normal'  # Normal label
    Label_tumor = 'Tumor'    # Tumor label

    Normal_images_label = 0 # Normal class
    Tumor_images_label = 1 # Tumor class

    Data_augmentation_normal = dataAugmentation(folder = Folder_normal, severity = Label_normal, sampling = Iter_normal, label = Normal_images_label, nfsave = False)
    Data_augmentation_tumor = dataAugmentation(folder = Folder_tumor, severity = Label_tumor, sampling = Iter_tumor, label = Tumor_images_label, nfsave = False)

    Images_Normal, Labels_Normal = Data_augmentation_normal.DataAugmentation()
    Images_Tumor, Labels_Tumor = Data_augmentation_tumor.DataAugmentation()

    Images.append(Images_Normal)
    Images.append(Images_Tumor)

    Labels.append(Labels_Normal)
    Labels.append(Labels_Tumor)

    return Images, Labels