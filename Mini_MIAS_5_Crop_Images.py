
import os
import cv2
import numpy as np
import pandas as pd

from Mini_MIAS_2_General_Functions import sort_images
from Mini_MIAS_2_General_Functions import remove_all_files

# ? class for images cropping.

class cropImages():

  def __init__(self, **kwargs):
    
    
    # * This algorithm outputs crop values for images based on the coordinates of the CSV file.
    # * General parameters
    self.Folder = kwargs.get('Folder', None)
    self.Normalfolder = kwargs.get('Normalfolder', None)
    self.Tumorfolder = kwargs.get('Tumorfolder', None)
    self.Benignfolder = kwargs.get('Benignfolder', None)
    self.Malignantfolder = kwargs.get('Malignantfolder', None)

    # * CSV to extract data
    self.Dataframe = kwargs.get('Dataframe', None)
    self.Shapes = kwargs.get('Shapes', None)
    
    # * X and Y mean to extract normal cropped images
    self.Xmean = kwargs.get('Xmean', None)
    self.Ymean = kwargs.get('Ymean', None)

    if self.Folder == None:
      raise ValueError("Folder does not exist") #! Alert

    elif self.Normalfolder == None:
      raise ValueError("Folder for normal images does not exist") #! Alert

    elif self.Tumorfolder == None:
      raise ValueError("Folder for tumor images does not exist") #! Alert

    elif self.Benignfolder == None:
      raise ValueError("Folder for benign images does not exist") #! Alert

    elif self.Malignantfolder == None:
      raise ValueError("Folder for malignant images does not exist") #! Alert

    #elif self.Dataframe == None:
      #raise ValueError("The dataframe is required") #! Alert

    elif self.Shapes == None:
      raise ValueError("The shape is required") #! Alert

    elif self.Xmean == None:
      raise ValueError("Xmean is required") #! Alert

    elif self.Ymean == None:
      raise ValueError("Ymean is required") #! Alert

  def CropMIAS(self):
    
    #Images = []

    os.chdir(self.Folder)

    # * Columns
    Name_column = 0
    Severity = 3
    X_column = 4
    Y_column = 5
    Radius = 6

    # * Labels
    Benign = 0
    Malignant = 1
    Normal = 2

    # * Initial index
    Index = 1
    
    # * Using sort function
    Sorted_files, Total_images = sort_images(self.Folder)
    Count = 1

    # * Reading the files
    for File in Sorted_files:
      
        Filename, Format = os.path.splitext(File)

        print("******************************************")
        print(self.Dataframe.iloc[Index - 1, Name_column])
        print(Filename)
        print("******************************************")

        if self.Dataframe.iloc[Index - 1, Severity] == Benign:
            if self.Dataframe.iloc[Index - 1, X_column] > 0  or self.Dataframe.iloc[Index - 1, Y_column] > 0:
              
                try:
                
                  print(f"Working with {Count} of {Total_images} {Format} Benign images, {Filename} X: {self.Dataframe.iloc[Index - 1, X_column]} Y: {self.Dataframe.iloc[Index - 1, Y_column]}")
                  print(self.Dataframe.iloc[Index - 1, Name_column], " ------ ", Filename, " ✅")
                  Count += 1

                  # * Reading the image
                  Path_file = os.path.join(self.Folder, File)
                  Image = cv2.imread(Path_file)
                
                  #Distance = self.Shape # X and Y.
                  #Distance = self.Shape # Perimetro de X y Y de la imagen.
                  #Image_center = Distance / 2 
                    
                  # * Obtaining the center using the radius
                  Image_center = self.Dataframe.iloc[Index - 1, Radius] / 2 
                  # * Obtaining dimension
                  Height_Y = Image.shape[0] 
                  print(Image.shape[0])
                  print(self.Dataframe.iloc[Index - 1, Radius])

                  # * Extract the value of X and Y of each image
                  X_size = self.Dataframe.iloc[Index - 1, X_column]
                  print(X_size)
                  Y_size = self.Dataframe.iloc[Index - 1, Y_column]
                  print(Y_size)
                    
                  # * Extract the value of X and Y of each image
                  XDL = X_size - Image_center
                  print(XDL)
                  XDM = X_size + Image_center
                  print(XDM)
                    
                  # * Extract the value of X and Y of each image
                  YDL = Height_Y - Y_size - Image_center
                  print(YDL)
                  YDM = Height_Y - Y_size + Image_center
                  print(YDM)

                  # * Cropped image
                  Cropped_Image_Benig = Image[int(YDL):int(YDM), int(XDL):int(XDM)]

                  print(Image.shape, " ----------> ", Cropped_Image_Benig.shape)

                  # print(Cropped_Image_Benig.shape)
                  # Display cropped image
                  # cv2_imshow(cropped_image)

                  New_name_filename = Filename + '_Benign_cropped' + Format

                  New_folder = os.path.join(self.Benignfolder, New_name_filename)
                  cv2.imwrite(New_folder, Cropped_Image_Benig)

                  New_folder = os.path.join(self.Tumorfolder, New_name_filename)
                  cv2.imwrite(New_folder, Cropped_Image_Benig)

                  #Images.append(Cropped_Image_Benig)

                except OSError:
                        print('Cannot convert %s' % File)

        elif self.Dataframe.iloc[Index - 1, Severity] == Malignant:
            if self.Dataframe.iloc[Index - 1, X_column] > 0  or self.Dataframe.iloc[Index - 1, Y_column] > 0:

                try:

                  print(f"Working with {Count} of {Total_images} {Format} Malignant images, {Filename} X {self.Dataframe.iloc[Index - 1, X_column]} Y {self.Dataframe.iloc[Index - 1, Y_column]}")
                  print(self.Dataframe.iloc[Index - 1, Name_column], " ------ ", Filename, " ✅")
                  Count += 1

                  # * Reading the image
                  Path_file = os.path.join(self.Folder, File)
                  Image = cv2.imread(Path_file)
                
                  #Distance = self.Shape # X and Y.
                  #Distance = self.Shape # Perimetro de X y Y de la imagen.
                  #Image_center = Distance / 2 
                    
                  # * Obtaining the center using the radius
                  Image_center = self.Dataframe.iloc[Index - 1, Radius] / 2 # Center
                  # * Obtaining dimension
                  Height_Y = Image.shape[0] 
                  print(Image.shape[0])

                  # * Extract the value of X and Y of each image
                  X_size = self.Dataframe.iloc[Index - 1, X_column]
                  Y_size = self.Dataframe.iloc[Index - 1, Y_column]
                    
                  # * Extract the value of X and Y of each image
                  XDL = X_size - Image_center
                  XDM = X_size + Image_center
                    
                  # * Extract the value of X and Y of each image
                  YDL = Height_Y - Y_size - Image_center
                  YDM = Height_Y - Y_size + Image_center

                  # * Cropped image
                  Cropped_Image_Malig = Image[int(YDL):int(YDM), int(XDL):int(XDM)]

                  print(Image.shape, " ----------> ", Cropped_Image_Malig.shape)
        
                  # print(Cropped_Image_Malig.shape)
                  # Display cropped image
                  # cv2_imshow(cropped_image)

                  New_name_filename = Filename + '_Malignant_cropped' + Format

                  New_folder = os.path.join(self.Malignantfolder, New_name_filename)
                  cv2.imwrite(New_folder, Cropped_Image_Malig)

                  New_folder = os.path.join(self.Tumorfolder, New_name_filename)
                  cv2.imwrite(New_folder, Cropped_Image_Malig)

                  #Images.append(Cropped_Image_Malig)


                except OSError:
                    print('Cannot convert %s' % File)
        
        elif self.Dataframe.iloc[Index - 1, Severity] == Normal:
          if self.Dataframe.iloc[Index - 1, X_column] == 0  or self.Dataframe.iloc[Index - 1, Y_column] == 0:

                try:

                  print(f"Working with {Count} of {Total_images} {Format} Normal images, {Filename}")
                  print(self.Dataframe.iloc[Index - 1, Name_column], " ------ ", Filename, " ✅")
                  Count += 1

                  Path_file = os.path.join(self.Folder, File)
                  Image = cv2.imread(Path_file)

                  Distance = self.Shapes # Perimetro de X y Y de la imagen.
                  Image_center = Distance / 2 # Centro de la imagen.
                  #CD = self.df.iloc[Index - 1, Radius] / 2
                  # * Obtaining dimension
                  Height_Y = Image.shape[0] 
                  print(Image.shape[0])

                  # * Extract the value of X and Y of each image
                  X_size = self.Xmean
                  Y_size = self.Ymean
                    
                  # * Extract the value of X and Y of each image
                  XDL = X_size - Image_center
                  XDM = X_size + Image_center
                    
                  # * Extract the value of X and Y of each image
                  YDL = Height_Y - Y_size - Image_center
                  YDM = Height_Y - Y_size + Image_center

                  # * Cropped image
                  Cropped_Image_Normal = Image[int(YDL):int(YDM), int(XDL):int(XDM)]

                  # * Comparison two images
                  print(Image.shape, " ----------> ", Cropped_Image_Normal.shape)

                  # print(Cropped_Image_Normal.shape)
                  # Display cropped image
                  # cv2_imshow(cropped_image)
              
                  New_name_filename = Filename + '_Normal_cropped' + Format

                  New_folder = os.path.join(self.Normalfolder, New_name_filename)
                  cv2.imwrite(New_folder, Cropped_Image_Normal)

                  #Images.append(Cropped_Image_Normal)

                except OSError:
                    print('Cannot convert %s' % File)

        Index += 1   

 
