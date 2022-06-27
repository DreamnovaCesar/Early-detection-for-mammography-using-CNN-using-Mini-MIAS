import cv2
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Normal
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Tumor
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Benign
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Malignant

from Mini_MIAS_3_Image_Processing import ImageProcessing

def preprocessing_Resize():

    # Parameters for resize

    X_new_size = 224
    Y_new_size = 224
    Interpolation = cv2.INTER_CUBIC

    # Image processing class

    Resize_normal_images = ImageProcessing(folder = Mini_MIAS_NT_Cropped_Images_Normal, Xresize = X_new_size, Yresize = Y_new_size, interpolation = Interpolation)
    Resize_tumor_images = ImageProcessing(folder = Mini_MIAS_NT_Cropped_Images_Tumor, Xresize = X_new_size, Yresize = Y_new_size, interpolation = Interpolation)
    Resize_benign_images = ImageProcessing(folder = Mini_MIAS_NT_Cropped_Images_Benign, Xresize = X_new_size, Yresize = Y_new_size, interpolation = Interpolation)
    Resize_malignant_images = ImageProcessing(folder = Mini_MIAS_NT_Cropped_Images_Malignant, Xresize = X_new_size, Yresize = Y_new_size, interpolation = Interpolation)

    # Image processing resize

    Resize_normal_images.resize_technique() # Resize normal images
    Resize_tumor_images.resize_technique() # Resize tumor images
    Resize_benign_images.resize_technique() # Resize benign images
    Resize_malignant_images.resize_technique() # Resize malignant images
