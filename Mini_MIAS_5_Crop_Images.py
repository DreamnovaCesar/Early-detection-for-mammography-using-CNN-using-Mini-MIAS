import os
import cv2
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from Mini_MIAS_2_General_Functions import SortImages
from Mini_MIAS_2_General_Functions import Removeallfiles

def MeanImages(df_M, column):

    """
	  Obtaining the mean value of the mammograms

    Parameters:
    argument1 (dataframe): dataframe that will be use to acquire the values
    argument2 (int): the column number to get the mean value

    Returns:
	  float:Returning the mean value
    """

    Data = []

    for i in range(df_M.shape[0]):
        if df_M.iloc[i - 1, column] > 0:
            Data.append(df_M.iloc[i - 1, column])

    Mean = int(np.mean(Data))

    print(Data)
    print(Mean)

    return Mean

def MIASCSV(CSV_Path):

    col_list = ["REFNUM", "BG", "CLASS", "SEVERITY", "X", "Y", "RADIUS"]
    df = pd.read_csv(CSV_Path, usecols = col_list)

    pd.set_option('display.max_rows', df.shape[0] + 1)
    print(df)

    df_T = CleaningMIASCSV(df)

    pd.set_option('display.max_rows', df_T.shape[0] + 1)
    print(df_T)

    return df_T

def CleaningMIASCSV(df_M):

    df_M.iloc[:, 3].values
    LE = LabelEncoder()
    df_M.iloc[:, 3] = LE.fit_transform(df_M.iloc[:, 3])

    df_M['X'] = df_M['X'].fillna(0)
    df_M['Y'] = df_M['Y'].fillna(0)
    df_M['RADIUS'] = df_M['RADIUS'].fillna(0)


    #df_M["X"].replace({"*NOTE": 0}, inplace = True)
    #df_M["Y"].replace({"3*": 0}, inplace = True)

    df_M['X'] = df_M['X'].astype(int)
    df_M['Y'] = df_M['Y'].astype(int)

    df_M['SEVERITY'] = df_M['SEVERITY'].astype(int)
    df_M['RADIUS'] = df_M['RADIUS'].astype(int)

    return df_M

# class for images cropping.

class cropImages():

  def __init__(self, **kwargs):
    
    self.folder = kwargs.get('folder', None)
    #self.newfolder = kwargs.get('foldersave', None)
    self.normalfolder = kwargs.get('normalfolder', None)
    self.tumorfolder = kwargs.get('tumorfolder', None)
    self.benignfolder = kwargs.get('benignfolder', None)
    self.malignantfolder = kwargs.get('malignantfolder', None)
    self.df = kwargs.get('df', None)
    self.shape = kwargs.get('shape', None)

    self.Xmean = kwargs.get('Xmean', None)
    self.Ymean = kwargs.get('Ymean', None)

  def CropMIAS(self):

    """

    Newfolder = r'\NT_Cropped_Images_'

    NormalFolder = self.newfolder + Newfolder + 'Normal'
    TumorFolder = self.newfolder + Newfolder + 'Tumor'
    BenignFolder = self.newfolder + Newfolder + 'Benign'
    MalignantFolder = self.newfolder + Newfolder + 'Malignant'

    ExistdirN = os.path.isdir(NormalFolder) 
    ExistdirT = os.path.isdir(TumorFolder)
    ExistdirB = os.path.isdir(BenignFolder) 
    ExistdirM = os.path.isdir(MalignantFolder) 

    if ExistdirN == False:
        Folder_Normal = os.path.join(self.newfolder, Newfolder + 'Normal')
        os.mkdir(Folder_Normal)

    if ExistdirT == False:
        Folder_Tumor = os.path.join(self.newfolder, Newfolder + 'Tumor')
        os.mkdir(Folder_Tumor)

    if ExistdirB == False:
        Folder_Benign = os.path.join(self.newfolder, Newfolder + 'Benign')
        os.mkdir(Folder_Benign)

    if ExistdirM == False:
        Folder_Malignant = os.path.join(self.newfolder, Newfolder + 'Malignant')
        os.mkdir(Folder_Malignant)
    """

    #Images = []

    os.chdir(self.folder)

    Refnum = 0
    Severity = 3
    Xcolumn = 4
    Ycolumn = 5
    Radius = 6

    Benign = 0
    Malignant = 1
    Normal = 2

    Index = 1

    png = ".png"    # png.

    sorted_files, images = SortImages(self.folder)
    count = 1
    k = 0

    for File in sorted_files:
      
        filename, extension  = os.path.splitext(File)

        print("******************************************")
        print(self.df.iloc[Index - 1, 0])
        print(filename)
        print("******************************************")

        if self.df.iloc[Index - 1, Severity] == Benign:
            if self.df.iloc[Index - 1, Xcolumn] > 0  or self.df.iloc[Index - 1, Ycolumn] > 0:
              
                try:
                
                    print(f"Working with {count} of {images} {extension} Benign images, {filename} X {self.df.iloc[Index - 1, Xcolumn]} Y {self.df.iloc[Index - 1, Ycolumn]}")
                    print(self.df.iloc[Index - 1, Refnum], " ------ ", filename, " ✅")
                    count += 1

                    Path_File = os.path.join(self.folder, File)
                    Imagen = cv2.imread(Path_File)
                
                    Distance = self.shape # X and Y.

                    #CD = Distance / 2 
                    CD = self.df.iloc[Index - 1, Radius] / 2 # Center
                    YA = Imagen.shape[0] # YAltura.

                    Xsize = self.df.iloc[Index - 1, Xcolumn]
                    Ysize = self.df.iloc[Index - 1, Ycolumn]
                    
                    XDL = Xsize - CD
                    XDM = Xsize + CD

                    YDL = YA - Ysize - CD
                    YDM = YA - Ysize + CD

                    # Cropped image
                    Cropped_Image_Benig = Imagen[int(YDL):int(YDM), int(XDL):int(XDM)]

                    print(Imagen.shape, " ----------> ", Cropped_Image_Benig.shape)

                    # print(Cropped_Image_Benig.shape)
                    # Display cropped image
                    # cv2_imshow(cropped_image)

                    dst_name = filename + '_Benign_cropped' + png

                    dstPath_name = os.path.join(self.benignfolder, dst_name)
                    cv2.imwrite(dstPath_name, Cropped_Image_Benig)

                    dstPath_name = os.path.join(self.tumorfolder, dst_name)
                    cv2.imwrite(dstPath_name, Cropped_Image_Benig)

                    #Images.append(Cropped_Image_Benig)

                except OSError:
                        print('Cannot convert %s' % File)

        elif self.df.iloc[Index - 1, Severity] == Malignant:
            if self.df.iloc[Index - 1, Xcolumn] > 0  or self.df.iloc[Index - 1, Ycolumn] > 0:

                try:

                  print(f"Working with {count} of {images} {extension} Malignant images, {filename} X {self.df.iloc[Index - 1, Xcolumn]} Y {self.df.iloc[Index - 1, Ycolumn]}")
                  print(self.df.iloc[Index - 1, Refnum], " ------ ", filename, " ✅")
                  count += 1

                  Path_File = os.path.join(self.folder, File)
                  Imagen = cv2.imread(Path_File)

                  Distance = self.shape # Perimetro de X y Y de la imagen.

                  #CD = Distance / 2 
                  CD = self.df.iloc[Index - 1, Radius] / 2
                  YA = Imagen.shape[0] # YAltura.

                  Xsize = self.df.iloc[Index - 1, Xcolumn]
                  Ysize = self.df.iloc[Index - 1, Ycolumn]

                  XDL = Xsize - CD
                  XDM = Xsize + CD

                  YDL = YA - Ysize - CD
                  YDM = YA - Ysize + CD

                  # Cropping an image
                  Cropped_Image_Malig = Imagen[int(YDL):int(YDM), int(XDL):int(XDM)]

                  print(Imagen.shape, " ----------> ", Cropped_Image_Malig.shape)

                  # print(Cropped_Image_Malig.shape)
                  # Display cropped image
                  # cv2_imshow(cropped_image)
              
                  dst_name = filename + '_Malignant_cropped' + png

                  dstPath_name = os.path.join(self.malignantfolder, dst_name)
                  cv2.imwrite(dstPath_name, Cropped_Image_Malig)

                  dstPath_name = os.path.join(self.tumorfolder, dst_name)
                  cv2.imwrite(dstPath_name, Cropped_Image_Malig)

                  #Images.append(Cropped_Image_Malig)


                except OSError:
                    print('Cannot convert %s' % File)
        
        elif self.df.iloc[Index - 1, Severity] == Normal:
          if self.df.iloc[Index - 1, Xcolumn] == 0  or self.df.iloc[Index - 1, Ycolumn] == 0:

                try:

                  print(f"Working with {count} of {images} {extension} Normal images, {filename}")
                  print(self.df.iloc[Index - 1, Refnum], " ------ ", filename, " ✅")
                  count += 1

                  Path_File = os.path.join(self.folder, File)
                  Imagen = cv2.imread(Path_File)

                  Distance = self.shape # Perimetro de X y Y de la imagen.
                  CD = Distance / 2 # Centro de la imagen.
                  #CD = self.df.iloc[Index - 1, Radius] / 2
                  YA = Imagen.shape[0] # YAltura.

                  Xsize = self.Xmean
                  Ysize = self.Ymean

                  XDL = Xsize - CD
                  XDM = Xsize + CD

                  YDL = YA - Ysize - CD
                  YDM = YA - Ysize + CD

                  # Cropping an image
                  Cropped_Image_Normal = Imagen[int(YDL):int(YDM), int(XDL):int(XDM)]

                  print(Imagen.shape, " ----------> ", Cropped_Image_Normal.shape)

                  # print(Cropped_Image_Malig.shape)
                  # Display cropped image
                  # cv2_imshow(cropped_image)
              
                  dst_name = filename + '_Normal_cropped' + png

                  dstPath_name = os.path.join(self.normalfolder, dst_name)
                  cv2.imwrite(dstPath_name, Cropped_Image_Normal)

                  #Images.append(Cropped_Image_Normal)

                except OSError:
                    print('Cannot convert %s' % File)

        Index += 1
        k += 1    

 
