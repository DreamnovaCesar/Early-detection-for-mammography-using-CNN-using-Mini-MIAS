
from Mini_MIAS_1_Folders import Mini_MIAS_PNG
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Normal
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Tumor
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Benign
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Malignant

from Mini_MIAS_2_General_Functions import mini_mias_csv_clean
from Mini_MIAS_2_General_Functions import extract_mean_from_images

from Mini_MIAS_5_Crop_Images import cropImages 

def preprocessing_Cropped_MIAS_Mammograms():

    x_column = 4
    y_column = 5
    cropped_images_shape = 112

    mias_csv = "D:\Mini-MIAS\Mini-MIAS Final\Mini_MIAS_CSV_DATA.csv"
    mias_csv_clean = mini_mias_csv_clean(mias_csv)

    x_mean = extract_mean_from_images(mias_csv_clean, x_column)
    y_mean = extract_mean_from_images(mias_csv_clean, y_column)

    MIAS = cropImages(  Folder = Mini_MIAS_PNG, 
                        Normalfolder = Mini_MIAS_NT_Cropped_Images_Normal, Tumorfolder = Mini_MIAS_NT_Cropped_Images_Tumor, 
                        Benignfolder = Mini_MIAS_NT_Cropped_Images_Benign, Malignantfolder = Mini_MIAS_NT_Cropped_Images_Malignant, 
                        Dataframe = mias_csv_clean, Shapes = cropped_images_shape, Xmean = x_mean, Ymean = y_mean )

    MIAS.CropMIAS()