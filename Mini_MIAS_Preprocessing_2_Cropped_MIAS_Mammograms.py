
from Mini_MIAS_1_Folders import ALLpng
from Mini_MIAS_1_Folders import NTCropped_Images_Normal
from Mini_MIAS_1_Folders import NTCropped_Images_Tumor
from Mini_MIAS_1_Folders import NTCropped_Images_Benign
from Mini_MIAS_1_Folders import NTCropped_Images_Malignant
#from Mini_MIAS_1_Folders import GeneralFolder

from Mini_MIAS_5_Crop_Images import MIASCSV
from Mini_MIAS_5_Crop_Images import MeanImages

from Mini_MIAS_5_Crop_Images import cropImages

def preprocessing_Cropped_MIAS_Mammograms():

    CSV_NIAS = "D:\Mini-MIAS\Mini-MIAS Final\Mini_MIAS_CSV_DATA.csv"
    Dataframe_MIAS = MIASCSV(CSV_NIAS)

    xmean = MeanImages(MIASdf, 4)
    ymean = MeanImages(MIASdf, 5)

    MIAS = cropImages(  folder = ALLpng, 
                        normalfolder = NTCropped_Images_Normal, tumorfolder = NTCropped_Images_Tumor, 
                        benignfolder = NTCropped_Images_Benign, malignantfolder = NTCropped_Images_Malignant, 
                        df = Dataframe_MIAS, shape = 112, Xmean = xmean, Ymean = ymean )

    MIAS.CropMIAS()