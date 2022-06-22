import cv2

from Mini_MIAS_1_Folders import NTCropped_Images_Normal
from Mini_MIAS_1_Folders import NTCropped_Images_Tumor
from Mini_MIAS_1_Folders import NTCropped_Images_Benign
from Mini_MIAS_1_Folders import NTCropped_Images_Malignant

from Mini_MIAS_3_Image_Processing import ImageProcessing

def preprocessing_Resize():

    # Parameters for resize

    XsizeResized = 224
    YsizeResized = 224
    interpolation = cv2.INTER_CUBIC

    # Image processing class

    MIASResize_N = ImageProcessing(folder = NTCropped_Images_Normal, Xresize = XsizeResized, Yresize = YsizeResized, interpolation = interpolation)
    MIASResize_T = ImageProcessing(folder = NTCropped_Images_Tumor, Xresize = XsizeResized, Yresize = YsizeResized, interpolation = interpolation)
    MIASResize_B = ImageProcessing(folder = NTCropped_Images_Benign, Xresize = XsizeResized, Yresize = YsizeResized, interpolation = interpolation)
    MIASResize_M = ImageProcessing(folder = NTCropped_Images_Malignant, Xresize = XsizeResized, Yresize = YsizeResized, interpolation = interpolation)

    # Image processing resize

    MIASResize_N.Resize() # Resize normal images
    MIASResize_T.Resize() # Resize tumor images
    MIASResize_B.Resize() # Resize benign images
    MIASResize_M.Resize() # Resize malignant images
