
from Mini_MIAS_1_Folders import Mini_MIAS_PNG
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Normal
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Tumor
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Benign
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Malignant

from Mini_MIAS_5_Crop_Images import mias_csv
from Mini_MIAS_5_Crop_Images import extract_mean_from_images

from Mini_MIAS_5_Crop_Images import cropImages

def preprocessing_Cropped_MIAS_Mammograms():

    x_column = 4
    y_column = 5
    cropped_images_shape = 112

    mini_mias_csv = "D:\Mini-MIAS\Mini-MIAS Final\Mini_MIAS_CSV_DATA.csv"
    mini_mias_csv_clean = mias_csv(mini_mias_csv)

    x_mean = extract_mean_from_images(mini_mias_csv_clean, x_column)
    y_mean = extract_mean_from_images(mini_mias_csv_clean, y_column)

    MIAS = cropImages(  folder = Mini_MIAS_PNG, 
                        normalfolder = Mini_MIAS_NT_Cropped_Images_Normal, tumorfolder = Mini_MIAS_NT_Cropped_Images_Tumor, 
                        benignfolder = Mini_MIAS_NT_Cropped_Images_Benign, malignantfolder = Mini_MIAS_NT_Cropped_Images_Malignant, 
                        df = mini_mias_csv_clean, shape = cropped_images_shape, Xmean = x_mean, Ymean = y_mean )

    MIAS.CropMIAS()