import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

# ? Detect fi GPU exist in your PC for CNN

def detect_GPU():

  # * This function shows if a gpu device is available and its name. 
  # * this is good to know if the training is using a GPU 

  GPU_name = tf.test.gpu_device_name()
  GPU_available = tf.test.is_gpu_available()

  #print(GPU_available)

  if GPU_available == True:
      print("GPU device is available")

  if "GPU" not in GPU_name:
      print("GPU device not found")
  print('Found GPU at: {}'.format(GPU_name))

# ? Sort Files

def sort_images(Folder_path): 

	"""
	Read all images in a folder and sort them.

    Parameters:
    argument1 (Folder): Folder used.

    Returns:
	  int:Returning value
    int:Returning list[str]

   	"""

  # * This function sort the files and show them

	Number_images = len(os.listdir(Folder_path))

	print("\n")
	print("********************************")
	print(f"Images: {Number_images}")
	print("********************************")
	print("\n")

	files = os.listdir(Folder_path)
	print(files)
	print("\n")

	print("********************************")
	Sorted_files =  sorted(files)
	print(Sorted_files)
	print("\n")
	print("********************************")

	return Sorted_files, Number_images

# ? Remove all files in folder

def remove_all_files(Folder_Path):

	"""
	Remove all images inside the folder chosen

    Parameters:
    argument1 (Folder): Folder used.

    Returns:
	Void

   	"""
  # * This function will remove all the files inside a folder

	for File in os.listdir(Folder_Path):
		filename, extension  = os.path.splitext(File)
		print(f"Removing {filename} {extension}✅")
		os.remove(os.path.join(Folder_Path, File))

# ? Extract the mean of each column

def extract_mean_from_images(Dataframe, Column):

    """
	  Obtaining the mean value of the mammograms

    Parameters:
    argument1 (dataframe): dataframe that will be use to acquire the values
    argument2 (int): the column number to get the mean value

    Returns:
	  float:Returning the mean value
    """
    # * This function will obtain the main of each column

    Data = []

    for i in range(Dataframe.shape[0]):
        if Dataframe.iloc[i - 1, Column] > 0:
            Data.append(Dataframe.iloc[i - 1, Column])

    Mean = int(np.mean(Data))

    #print(Data)
    #print(Mean)

    return Mean

# ? Clean Mini-MIAS CSV

def mini_mias_csv_clean(Dataframe):

    # * This function will clean the data from the CSV archive

    col_list = ["REFNUM", "BG", "CLASS", "SEVERITY", "X", "Y", "RADIUS"]
    Dataframe_Mini_MIAS = pd.read_csv(Dataframe, usecols = col_list)

    # * Severity's column
    Mini_MIAS_severity_column = 3

    # * it labels each severity grade

    Dataframe_Mini_MIAS.iloc[:, Mini_MIAS_severity_column].values
    LE = LabelEncoder()
    Dataframe_Mini_MIAS.iloc[:, Mini_MIAS_severity_column] = LE.fit_transform(Dataframe_Mini_MIAS.iloc[:, 3])

    # * Fullfill X, Y and RADIUS columns with 0
    Dataframe_Mini_MIAS['X'] = Dataframe_Mini_MIAS['X'].fillna(0)
    Dataframe_Mini_MIAS['Y'] = Dataframe_Mini_MIAS['Y'].fillna(0)
    Dataframe_Mini_MIAS['RADIUS'] = Dataframe_Mini_MIAS['RADIUS'].fillna(0)

    #Dataframe["X"].replace({"*NOTE": 0}, inplace = True)
    #Dataframe["Y"].replace({"3*": 0}, inplace = True)

    # * X and Y columns tranform into int type
    Dataframe_Mini_MIAS['X'] = Dataframe_Mini_MIAS['X'].astype(int)
    Dataframe_Mini_MIAS['Y'] = Dataframe_Mini_MIAS['Y'].astype(int)

    # * Severity and radius columns tranform into int type
    Dataframe_Mini_MIAS['SEVERITY'] = Dataframe_Mini_MIAS['SEVERITY'].astype(int)
    Dataframe_Mini_MIAS['RADIUS'] = Dataframe_Mini_MIAS['RADIUS'].astype(int)

    return Dataframe_Mini_MIAS

# ? Concat multiple dataframes

def concat_dataframe(*dfs, **kwargs):

    # * this function concatenate the number of dataframes added

    # * General parameters

    folder = kwargs.get('folder', None)
    label = kwargs.get('label', None)
    technique = kwargs.get('technique', None)

    if folder == None:
      raise ValueError("Folder does not exist") #! Alert

    elif label == None:
      raise ValueError("Class does not exist")  #! Alert

    elif technique == None:
      raise ValueError("Technique does not exist")  #! Alert

    # * Concatenate each dataframe
    ALL_dataframes = [df for df in dfs]

    Final_dataframe = pd.concat(ALL_dataframes, ignore_index = True, sort = False)
        
    #pd.set_option('display.max_rows', Final_dataframe.shape[0] + 1)
    #print(DataFrame)

    # * Name the final dataframe and save it into the given path
    Name_dataframe =  str(label) + '_Dataframe_' + str(technique) + '.csv'
    Folder_dataframe_to_save = os.path.join(folder, Name_dataframe)

    Final_dataframe.to_csv(Folder_dataframe_to_save)

# ? Transform initial format to another. (PGM to PNG) / (PGM to TIFF)

class changeFormat:
  
  # * Change the format of one image to another 

  def __init__(self, **kwargs):
    
    # * General parameters
    self.Folder = kwargs.get('Folder', None)
    self.Newfolder = kwargs.get('Newfolder', None)
    self.Format = kwargs.get('Format', None)
    self.Newformat = kwargs.get('Newformat', None)

    if self.Folder == None:
      raise ValueError("Folder does not exist") #! Alert

    elif self.Newfolder == None:
      raise ValueError("Folder destination does not exist") #! Alert

    elif self.Format == None:
      raise ValueError("format has to be added") #! Alert

    elif self.Newformat == None:
      raise ValueError("New format has to be added") #! Alert

  def ChangeExtension(self):

    # * Changes the current working directory to the given path
    os.chdir(self.folder)

    print(os.getcwd())
    
    # * Using the sort function
    Sorted_files, Total_images = sort_images(self.folder)
    Count = 0

    # * Reading the files
    for File in Sorted_files:
      if File.endswith(self.format):

        try:
            Filename, format  = os.path.splitext(File)
            print(f"Working with {Count} of {Total_images} {self.format} images, {Filename} ------- {self.newformat} ✅")
            Count += 1
            
            # * Reading each image using cv2
            Path_file = os.path.join(self.folder, File)
            Imagen = cv2.imread(Path_file)         
            #Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)
            
            # * Changing its format to a new one
            New_name_filename = Filename + self.newformat
            New_folder = os.path.join(self.newfolder, New_name_filename)

            cv2.imwrite(New_folder, Imagen)
            #FilenamesREFNUM.append(Filename)

        except OSError:
            print('Cannot convert %s ❌' % File) #! Alert

    print("\n")
    print(f"COMPLETE {Count} of {Total_images} TRANSFORMED ✅")
