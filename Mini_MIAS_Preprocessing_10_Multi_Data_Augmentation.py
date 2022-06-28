
from Mini_MIAS_4_Data_Augmentation import dataAugmentation

def preprocessing_DataAugmentation_Multiclass(Folder_normal, Folder_benign, Folder_malignant):

    # * List to add images and labels.
    Images = []
    Labels = []

    # * General parameters
    Iter_normal = 2
    Iter_benign = 70
    Iter_malignant = 90 

    Label_bormal = 'Normal'
    Label_benign = 'Benign'
    Label_malignant = 'Malignant' 

    Normal_images_class = 0
    Benign_images_class = 1
    Malignant_images_class = 2 

    # * With this class we use the technique called data augmentation to create new images with their transformations
    Data_augmentation_normal = dataAugmentation(folder = Folder_normal, severity = Label_bormal, sampling = Iter_normal, label = Normal_images_class, nfsave = False)
    Data_augmentation_benign = dataAugmentation(folder = Folder_benign, severity = Label_benign, sampling = Iter_benign, label = Benign_images_class, nfsave = False)
    Data_augmentation_malignant = dataAugmentation(folder = Folder_malignant, severity = Label_malignant, sampling = Iter_malignant, label = Malignant_images_class, nfsave = False)

    Images_Normal, Labels_Normal = Data_augmentation_normal.data_augmentation()
    Images_Benign, Labels_Benign = Data_augmentation_benign.data_augmentation()
    Images_Malignant, Labels_Malignant = Data_augmentation_malignant.data_augmentation()

    # * Add the valueS in the lists already created
    Images.append(Images_Normal)
    Images.append(Images_Benign)
    Images.append(Images_Malignant)

    Labels.append(Labels_Normal)
    Labels.append(Labels_Benign)
    Labels.append(Labels_Malignant)

    return Images, Labels