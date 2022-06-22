import os
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from Mini_MIAS_7_CNN_Architectures import PreTrainedModels

# Sort Files

def SortImages(Folder_Path): 

	"""
	Read all images in a folder and sort them.

    Parameters:
    argument1 (Folder): Folder used.

    Returns:
	int:Returning value
    int:Returning list[str]

   	"""

	NumberImages = len(os.listdir(Folder_Path))

	print("\n")
	print("********************************")
	print(f"Images: {NumberImages}")
	print("********************************")
	print("\n")

	files = os.listdir(Folder_Path)
	print(files)
	print("\n")

	print("********************************")
	sorted_files =  sorted(files)
	print(sorted_files)
	print("\n")
	print("********************************")

	return sorted_files, NumberImages

# Remove all files in folder

def Removeallfiles(Folder_Path):

	"""
	Remove all images inside the folder chosen

    Parameters:
    argument1 (Folder): Folder used.

    Returns:
	Void

   	"""

	for File in os.listdir(Folder_Path):
		filename, extension  = os.path.splitext(File)
		print(f"Removing {filename} ✅")
		os.remove(os.path.join(Folder_Path, File))

class changeExtension:
  
  def __init__(self, **kwargs):
    
    self.folder = kwargs.get('folder', None)
    self.newfolder = kwargs.get('newfolder', None)
    self.extension = kwargs.get('extension', None)
    self.newextension = kwargs.get('newextension', None)

    if self.folder == None:
      raise ValueError("Folder does not exist")

    elif self.newfolder == None:
      raise ValueError("Destination Folder does not exist")

    elif self.extension == None:
      raise ValueError("Extension does not exist")

    elif self.newextension == None:
      raise ValueError("New extension does not exist")

  def ChangeExtension(self):

    os.chdir(self.folder)

    print(os.getcwd())

    sorted_files, images = SortImages(self.folder)
    count = 0
  
    for File in sorted_files:
      if File.endswith(self.extension): # Read png files

        try:
            filename, extension  = os.path.splitext(File)
            print(f"Working with {count} of {images} {self.extension} images, {filename} ------- {self.newextension} ✅")
            count += 1
            
            Path_File = os.path.join(self.folder, File)
            Imagen = cv2.imread(Path_File)         
            #Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)
            
            dst_name = filename + self.newextension
            dstPath_name = os.path.join(self.newfolder, dst_name)

            cv2.imwrite(dstPath_name, Imagen)
            #FilenamesREFNUM.append(filename)

        except OSError:
            print('Cannot convert %s ❌' % File)

    print("\n")
    print(f"COMPLETE {count} of {images} TRANSFORMED ✅")

# Concat multiple dataframes

def DataframeSave(*dfs, **kwargs):

    folder = kwargs.get('folder', None)
    Class = kwargs.get('Class', None)
    technique = kwargs.get('technique', None)

    if folder == None:
      raise ValueError("Folder does not exist")

    elif Class == None:
      raise ValueError("Class does not exist")

    elif technique == None:
      raise ValueError("Technique does not exist")

    ALLdf = [df for df in dfs]

    DataFrame = pd.concat(ALLdf, ignore_index = True, sort = False)
        
    pd.set_option('display.max_rows', DataFrame.shape[0] + 1)
    #print(DataFrame)

    dst =  str(Class) + '_Dataframe_' + str(technique) + '.csv'
    dstPath = os.path.join(folder, dst)

    DataFrame.to_csv(dstPath)

# Configuration of each DCNN model

def ConfigurationModels(MainKeys, Arguments, Folder_Save, Folder_Save_Esp):

    TotalImage = []
    TotalLabel = []

    ClassSize = (len(Arguments[2]))
    Images = 7
    Labels = 8

    if len(Arguments) == len(MainKeys):
        
        DicAruments = dict(zip(MainKeys, Arguments))

        for i in range(ClassSize):

            for element in list(DicAruments.values())[Images]:
                TotalImage.append(element)
            
            #print('Total:', len(TotalImage))
        
            for element in list(DicAruments.values())[Labels]:
                TotalLabel.append(element)

            #print('Total:', len(TotalLabel))

            Images += 2
            Labels += 2

        #TotalImage = [*list(DicAruments.values())[Images], *list(DicAruments.values())[Images + 2]]
        
    elif len(Arguments) > len(MainKeys):

        TotalArguments = len(Arguments) - len(MainKeys)

        for i in range(TotalArguments // 2):

            MainKeys.append('Images ' + str(i + 3))
            MainKeys.append('Labels ' + str(i + 3))

        DicAruments = dict(zip(MainKeys, Arguments))

        for i in range(ClassSize):

            for element in list(DicAruments.values())[Images]:
                TotalImage.append(element)
            
            for element in list(DicAruments.values())[Labels]:
                TotalLabel.append(element)

            Images += 2
            Labels += 2

    elif len(Arguments) < len(MainKeys):

        raise ValueError('its impossible')

    #print(DicAruments)

    def printDict(DicAruments):

        for i in range(7):
            print(list(DicAruments.items())[i])

    printDict(DicAruments)

    print(len(TotalImage))
    print(len(TotalLabel))

    X_train, X_test, y_train, y_test = train_test_split(np.array(TotalImage), np.array(TotalLabel), test_size = 0.20, random_state = 42)

    # convert from integers to floats
    #X_train = X_train.astype('float32')
    #X_test = X_test.astype('float32')
    # normalize to range 0-1
    #X_train = X_train / 255.0
    #X_test = X_test / 255.0

    Score = PreTrainedModels(Arguments[0], Arguments[1], Arguments[2], Arguments[3], Arguments[4], ClassSize, Arguments[5], Arguments[6], X_train, y_train, X_test, y_test, Folder_Save, Folder_Save_Esp)
    #Score = PreTrainedModels(ModelPreTrained, technique, labels, Xsize, Ysize, num_classes, vali_split, epochs, X_train, y_train, X_test, y_test)
    return Score

# Update CSV changing value

def UpdateCSV(Score, df, column_names, path, row):

    """
	  Printing amount of images with data augmentation 

    Parameters:
    argument1 (list): The number of Normal images.
    argument2 (list): The number of Tumor images.
    argument3 (str): Technique used

    Returns:
	  void
   	"""
     
    for i in range(len(Score)):
        df.loc[row, column_names[i]] = Score[i]
  
    df.to_csv(path, index = False)
  
    print(df)

# Transform initial format to another. (PGM to PNG) / (PGM to TIFF)

class changeExtension:
  
  def __init__(self, **kwargs):
    
    self.folder = kwargs.get('folder', None)
    self.newfolder = kwargs.get('newfolder', None)
    self.extension = kwargs.get('extension', None)
    self.newextension = kwargs.get('newextension', None)

    if self.folder == None:
      raise ValueError("Folder does not exist")

    elif self.newfolder == None:
      raise ValueError("Destination Folder does not exist")

    elif self.extension == None:
      raise ValueError("Extension does not exist")

    elif self.newextension == None:
      raise ValueError("New extension does not exist")

  def ChangeExtension(self):

    os.chdir(self.folder)

    print(os.getcwd())

    sorted_files, images = SortImages(self.folder)
    count = 0
  
    for File in sorted_files:
      if File.endswith(self.extension): # Read png files

        try:
            filename, extension  = os.path.splitext(File)
            print(f"Working with {count} of {images} {self.extension} images, {filename} ------- {self.newextension} ✅")
            count += 1
            
            Path_File = os.path.join(self.folder, File)
            Imagen = cv2.imread(Path_File)         
            #Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)
            
            dst_name = filename + self.newextension
            dstPath_name = os.path.join(self.newfolder, dst_name)

            cv2.imwrite(dstPath_name, Imagen)
            #FilenamesREFNUM.append(filename)

        except OSError:
            print('Cannot convert %s ❌' % File)

    print("\n")
    print(f"COMPLETE {count} of {images} TRANSFORMED ✅")
