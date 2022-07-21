import cv2
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Normal
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Tumor
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Benign
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Malignant

from Mini_MIAS_3_Image_Processing import ImageProcessing

def preprocessing_Resize(Folder_normal, Folder_tumor, Folder_benign, Folder_malignant):

    # * Parameters for resize

    X_new_size = 224
    Y_new_size = 224
    Interpolation = cv2.INTER_CUBIC

    # * Image processing class

    Resize_normal_images = ImageProcessing(folder = Folder_normal, Xresize = X_new_size, Yresize = Y_new_size, interpolation = Interpolation)
    Resize_tumor_images = ImageProcessing(folder = Folder_tumor, Xresize = X_new_size, Yresize = Y_new_size, interpolation = Interpolation)
    Resize_benign_images = ImageProcessing(folder = Folder_benign, Xresize = X_new_size, Yresize = Y_new_size, interpolation = Interpolation)
    Resize_malignant_images = ImageProcessing(folder = Folder_malignant, Xresize = X_new_size, Yresize = Y_new_size, interpolation = Interpolation)

    # * Image processing resize

    Resize_normal_images.resize_technique() 
    Resize_tumor_images.resize_technique() 
    Resize_benign_images.resize_technique() 
    Resize_malignant_images.resize_technique() 
