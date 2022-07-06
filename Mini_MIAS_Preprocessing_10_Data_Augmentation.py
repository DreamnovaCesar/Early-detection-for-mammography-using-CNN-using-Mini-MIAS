
from Mini_MIAS_4_Data_Augmentation import dataAugmentation

def preprocessing_DataAugmentation_Biclass(Folder_normal, Folder_tumor):

    # * List to add images and labels.
    Images = []
    Labels = []

    # * General parameters
    #Iter_normal = 20 
    #Iter_tumor = 40 

    Iter_normal = 3 
    Iter_tumor = 9 

    Label_normal = 'Normal' 
    Label_tumor = 'Tumor'  

    Normal_images_class = 0 
    Tumor_images_class = 1 

    # * With this class we use the technique called data augmentation to create new images with their transformations
    Data_augmentation_normal = dataAugmentation(Folder = Folder_normal, Severity = Label_normal, Sampling = Iter_normal, Label = Normal_images_class, Saveimages = False)
    Data_augmentation_tumor = dataAugmentation(Folder = Folder_tumor, Severity = Label_tumor, Sampling = Iter_tumor, Label = Tumor_images_class, Saveimages = False)

    Images_Normal, Labels_Normal = Data_augmentation_normal.data_augmentation()
    Images_Tumor, Labels_Tumor = Data_augmentation_tumor.data_augmentation()

    # * Add the value in the lists already created
    Images.append(Images_Normal)
    Images.append(Images_Tumor)

    Labels.append(Labels_Normal)
    Labels.append(Labels_Tumor)

    return Images, Labels