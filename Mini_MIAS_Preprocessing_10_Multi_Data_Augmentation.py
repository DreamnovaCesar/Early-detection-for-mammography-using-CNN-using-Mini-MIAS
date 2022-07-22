import numpy as np

from Mini_MIAS_4_Data_Augmentation import dataAugmentation

def preprocessing_DataAugmentation_Multiclass_ML(Folder_normal, Folder_benign, Folder_malignant, Folder_destination):

    # * List to add images and labels.
    Images = []
    Labels = []

    # * General parameters
    #Iter_normal = 2
    #Iter_benign = 70
    #Iter_malignant = 90 

    Iter_normal = 7
    Iter_benign = 25
    Iter_malignant = 35 

    #Iter_normal = 2
    #Iter_benign = 8
    #Iter_malignant = 10 

    Label_bormal = 'Normal'
    Label_benign = 'Benign'
    Label_malignant = 'Malignant' 

    Normal_images_class = 0
    Benign_images_class = 1
    Malignant_images_class = 2 

    # * With this class we use the technique called data augmentation to create new images with their transformations
    Data_augmentation_normal = dataAugmentation(Folder = Folder_normal, NewFolder = Folder_destination, Severity = Label_bormal, Sampling = Iter_normal, Label = Normal_images_class, Saveimages = False)
    Data_augmentation_benign = dataAugmentation(Folder = Folder_benign, NewFolder = Folder_destination, Severity = Label_benign, Sampling = Iter_benign, Label = Benign_images_class, Saveimages = False)
    Data_augmentation_malignant = dataAugmentation(Folder = Folder_malignant, NewFolder = Folder_destination, Severity = Label_malignant, Sampling = Iter_malignant, Label = Malignant_images_class, Saveimages = False)

    Images_Normal, Labels_Normal = Data_augmentation_normal.data_augmentation()
    Images_Benign, Labels_Benign = Data_augmentation_benign.data_augmentation()
    Images_Malignant, Labels_Malignant = Data_augmentation_malignant.data_augmentation()

    # * Add the value in the lists already created

    Images.append(Images_Normal)
    Images.append(Images_Benign)
    Images.append(Images_Malignant)

    Labels.append(Labels_Normal)
    Labels.append(Labels_Benign)
    Labels.append(Labels_Malignant)
    
    print(len(Images_Normal))
    print(len(Images_Benign))
    print(len(Images_Malignant))

    return Images, Labels

def preprocessing_DataAugmentation_Multiclass_CNN(Folder_normal, Folder_benign, Folder_malignant, Folder_destination):

    # * List to add images and labels.
    #Images = []
    #Labels = []

    # * General parameters
    #Iter_normal = 2
    #Iter_benign = 70
    #Iter_malignant = 90 

    Iter_normal = 7
    Iter_benign = 25
    Iter_malignant = 35 

    #Iter_normal = 2
    #Iter_benign = 8
    #Iter_malignant = 10 

    Label_bormal = 'Normal'
    Label_benign = 'Benign'
    Label_malignant = 'Malignant' 

    Normal_images_class = 0
    Benign_images_class = 1
    Malignant_images_class = 2 

    # * With this class we use the technique called data augmentation to create new images with their transformations
    Data_augmentation_normal = dataAugmentation(Folder = Folder_normal, NewFolder = Folder_destination, Severity = Label_bormal, Sampling = Iter_normal, Label = Normal_images_class, Saveimages = True)
    Data_augmentation_benign = dataAugmentation(Folder = Folder_benign, NewFolder = Folder_destination, Severity = Label_benign, Sampling = Iter_benign, Label = Benign_images_class, Saveimages = True)
    Data_augmentation_malignant = dataAugmentation(Folder = Folder_malignant, NewFolder = Folder_destination, Severity = Label_malignant, Sampling = Iter_malignant, Label = Malignant_images_class, Saveimages = True)

    Images_Normal, Labels_Normal = Data_augmentation_normal.data_augmentation()
    Images_Benign, Labels_Benign = Data_augmentation_benign.data_augmentation()
    Images_Malignant, Labels_Malignant = Data_augmentation_malignant.data_augmentation()

    # * Add the value in the lists already created

    Images_total = Images_Normal + Images_Benign + Images_Malignant
    Labels_total = np.concatenate((Labels_Normal, Labels_Benign, Labels_Malignant), axis = None)
    
    print(len(Images_Normal))
    print(len(Images_Benign))
    print(len(Images_Malignant))

    return Images_total, Labels_total

