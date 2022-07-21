import os
import cv2
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2s

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import normalized_mutual_information as nmi

from skimage import io
from skimage import filters
from skimage import img_as_ubyte
from skimage import img_as_float

from skimage.exposure import equalize_adapthist
from skimage.exposure import equalize_hist
from skimage.exposure import rescale_intensity

from skimage.filters import unsharp_mask

from Mini_MIAS_2_General_Functions import sort_images
from Mini_MIAS_2_General_Functions import remove_all_files

class ImageProcessing:

  def __init__(self, **kwargs):

    # * General parameters
    self.Folder = kwargs.get('Folder', None)
    self.Newfolder = kwargs.get('Newfolder', None)
    self.Severity = kwargs.get('Severity', None)
    self.Label = kwargs.get('Label', None)

    # * Parameters for resizing
    self.Interpolation = kwargs.get('Interpolation', cv2.INTER_CUBIC)
    self.Xresize = kwargs.get('Xresize', 224)
    self.Yresize = kwargs.get('Yresize', 224)

    # * Parameters for median filter
    self.Division = kwargs.get('Division', 3)

    # * Parameters for CLAHE
    self.Cliplimit = kwargs.get('Cliplimit', 0.01)

    # * Parameters for unsharp masking
    self.Radius = kwargs.get('Radius', 1)
    self.Amount = kwargs.get('Amount', 1)

    if self.Folder == None:
      raise ValueError("Folder does not exist") #! Alert

    #if self.Newfolder == None:
      #raise ValueError("Folder destination does not exist") #! Alert

    #if self.Severity == None:
      #raise ValueError("Assign the severity") #! Alert

    #if self.Label == None:
      #raise ValueError("Assign the interpolation that will be used") #! Alert

  # ? Resize technique method

  def resize_technique(self):

    # * Save the new images in a list
    #New_images = [] 

    os.chdir(self.Folder)
    print(os.getcwd())
    print("\n")

    # * Using sort function
    Sorted_files, Total_images = sort_images(self.Folder)
    count = 1

    # * Reading the files
    for File in Sorted_files:
      
      # * Extract the file name and format
      Filename, Format  = os.path.splitext(File)
      
      if File.endswith(Format):

        try:
          print(f"Working with {count} of {Total_images} normal images")
          count += 1
          
          # * Resize with the given values 
          Path_file = os.path.join(self.Folder, File)
          Imagen = cv2.imread(Path_file)

          # * Resize with the given values 
          Shape = (self.Xresize, self.Yresize)
          Resized_Imagen = cv2.resize(Imagen, Shape, interpolation = self.Interpolation)

          # * Show old image and new image
          print(Imagen.shape, ' -------- ', Resized_Imagen.shape)

          # * Name the new file
          New_name_filename = Filename + Format
          New_folder = os.path.join(self.Folder, New_name_filename)

          # * Save the image in a new folder
          cv2.imwrite(New_folder, Resized_Imagen)
          #New_images.append(Resized_Imagen)
          
        except OSError:
          print('Cannot convert %s ❌' % File) #! Alert

    print("\n")
    print(f"COMPLETE {count} of {Total_images} RESIZED ✅")

  # ? Normalization technique method

  def normalize_technique(self):

    """
	  Get the values from median filter images and save them into a dataframe.

    Parameters:
    argument1 (Folder): Folder chosen.
    argument2 (Folder): Folder destination for new images.
    argument3 (Str): Severity of each image.
    argument4 (Int): The number of the images.
    argument5 (Int): Division for median filter.
    argument6 (Str): Label for each image.

    Returns:
	  int:Returning dataframe with all data.
    
    """
    # * Remove all the files in the new folder using this function
    remove_all_files(self.Newfolder)

    # * Lists to save the values of the labels and the filename and later use them for a dataframe
    #Images = [] 
    Labels = []
    All_filenames = []
    
    # * Lists to save the values of each statistic
    Mae_ALL = [] # ? Mean Absolute Error.
    Mse_ALL = [] # ? Mean Squared Error.
    Ssim_ALL = [] # ? Structural similarity.
    Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
    Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
    Nmi_ALL = [] # ? Normalized Mutual Information.
    R2s_ALL = [] # ? Coefficient of determination.

    os.chdir(self.Folder)

    # * Using sort function
    Sorted_files, Total_images = sort_images(self.Folder)
    Count = 1

    # * Reading the files
    for File in Sorted_files:

      # * Extract the file name and format
      Filename, Format  = os.path.splitext(File)

      # * Read extension files
      if File.endswith(Format): 
        
        print(f"Working with {Count} of {Total_images} {self.Severity} images ✅")
        print(f"Working with {Filename} ✅")

        # * Resize with the given values
        Path_file = os.path.join(self.Folder, File)
        Image = cv2.imread(Path_file)

        Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

        #print("%s has %d and %d" % (File, Imagen.shape[0], Imagen.shape[1]))

        # * Add a black image for normalization
        Norm_img = np.zeros((Image.shape[0], Image.shape[1]))
        Normalization_Imagen = cv2.normalize(Image, Norm_img, 0, 255, cv2.NORM_MINMAX)

        # * Save each statistic in a variable
        Mae = mae(Image, Normalization_Imagen)
        Mse = mse(Image, Normalization_Imagen)
        Ssim = ssim(Image, Normalization_Imagen)
        Psnr = psnr(Image, Normalization_Imagen)
        Nrmse = nrmse(Image, Normalization_Imagen)
        Nmi = nmi(Image, Normalization_Imagen)
        R2s = r2s(Image, Normalization_Imagen)

        # * Add the value in the lists already created
        Mae_ALL.append(Mae)
        Mse_ALL.append(Mse)
        Ssim_ALL.append(Ssim)
        Psnr_ALL.append(Psnr)
        Nrmse_ALL.append(Nrmse)
        Nmi_ALL.append(Nmi)
        R2s_ALL.append(R2s)

        # * Name the new file
        Filename_and_technique = Filename + '_Normalization'
        New_name_filename = Filename_and_technique + Format
        New_folder = os.path.join(self.Newfolder, New_name_filename)
        
        #Normalization_Imagen = Normalization_Imagen.astype('float32')
        #Normalization_Imagen = Normalization_Imagen / 255.0
        #print(Normalization_Imagen)

        # * Save the image in a new folder
        cv2.imwrite(New_folder, Normalization_Imagen)
        
        # * Save the values of labels and each filenames
        #Images.append(Normalization_Imagen)
        Labels.append(self.Label)
        All_filenames.append(Filename_and_technique)

        Count += 1
    
    # * Return the new dataframe with the new data
    Dataframe = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

    return Dataframe

  # ? Median filter technique method

  def median_filter_technique(self):

      """
      Get the values from median filter images and save them into a dataframe.

      Parameters:
      argument1 (Folder): Folder chosen.
      argument2 (Folder): Folder destination for new images.
      argument3 (Str): Severity of each image.
      argument4 (Int): The number of the images.
      argument5 (Int): Division for median filter.
      argument6 (Str): Label for each image.

      Returns:
      int:Returning dataframe with all data.
      
      """
      # * Remove all the files in the new folder using this function
      remove_all_files(self.Newfolder)
      
      # * Lists to save the values of the labels and the filename and later use them for a dataframe
      #Images = [] 
      Labels = []
      All_filenames = []

      # * Lists to save the values of each statistic
      Mae_ALL = [] # ? Mean Absolute Error.
      Mse_ALL = [] # ? Mean Squared Error.
      Ssim_ALL = [] # ? Structural similarity.
      Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
      Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
      Nmi_ALL = [] # ? Normalized Mutual Information.
      R2s_ALL = [] # ? Coefficient of determination.

      os.chdir(self.Folder)

      # * Using sort function
      Sorted_files, Total_images = sort_images(self.Folder)
      Count = 1

      # * Reading the files
      for File in Sorted_files:
        
        # * Extract the file name and format
        Filename, Format  = os.path.splitext(File)

        # * Read extension files
        if File.endswith(Format):
          
          print(f"Working with {Count} of {Total_images} {self.Severity} images ✅")
          print(f"Working with {Filename} ✅")

          # * Resize with the given values
          Path_file = os.path.join(self.Folder, File)
          Image = io.imread(Path_file, as_gray = True)

          #Image_median_filter = cv2.medianBlur(Imagen, Division)
          Median_filter_image = filters.median(Image, np.ones((self.Division, self.Division)))

          # * Save each statistic in a variable
          Mae = mae(Image, Median_filter_image)
          Mse = mse(Image, Median_filter_image)
          Ssim = ssim(Image, Median_filter_image)
          Psnr = psnr(Image, Median_filter_image)
          Nrmse = nrmse(Image, Median_filter_image)
          Nmi = nmi(Image, Median_filter_image)
          R2s = r2s(Image, Median_filter_image)

          # * Add the value in the lists already created
          Mae_ALL.append(Mae)
          Mse_ALL.append(Mse)
          Ssim_ALL.append(Ssim)
          Psnr_ALL.append(Psnr)
          Nrmse_ALL.append(Nrmse)
          Nmi_ALL.append(Nmi)
          R2s_ALL.append(R2s)

          # * Name the new file
          Filename_and_technique = Filename + '_Median_Filter'
          New_name_filename = Filename_and_technique + Format
          New_folder = os.path.join(self.Newfolder, New_name_filename)

          # * Save the image in a new folder
          io.imsave(New_folder, Median_filter_image)

          # * Save the values of labels and each filenames
          #Images.append(Normalization_Imagen)
          Labels.append(self.Label)
          All_filenames.append(Filename_and_technique)

          Count += 1

      # * Return the new dataframe with the new data
      DataFrame = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

      return DataFrame

  # ? CLAHE technique method

  def CLAHE_technique(self):

      """
      Get the values from CLAHE images and save them into a dataframe.

      Parameters:
      argument1 (Folder): Folder chosen.
      argument2 (Folder): Folder destination for new images.
      argument3 (Str): Severity of each image.
      argument4 (Float): clip limit value use to change CLAHE images.
      argument5 (Str): Label for each image.

      Returns:
      int:Returning dataframe with all data.
      
      """
      # * Remove all the files in the new folder using this function
      remove_all_files(self.Newfolder)
      
      # * Lists to save the values of the labels and the filename and later use them for a dataframe
      #Images = [] 
      Labels = []
      All_filenames = []

      # * Lists to save the values of each statistic
      Mae_ALL = [] # ? Mean Absolute Error.
      Mse_ALL = [] # ? Mean Squared Error.
      Ssim_ALL = [] # ? Structural similarity.
      Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
      Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
      Nmi_ALL = [] # ? Normalized Mutual Information.
      R2s_ALL = [] # ? Coefficient of determination.

      os.chdir(self.Folder)

      # * Using sort function
      Sorted_files, Total_images = sort_images(self.Folder)
      Count = 1

      # * Reading the files
      for File in Sorted_files:
        
        # * Extract the file name and format
        Filename, Format  = os.path.splitext(File)

        # * Read extension files
        if File.endswith(Format):
          
          print(f"Working with {Count} of {Total_images} {self.Severity} images ✅")
          print(f"Working with {Filename} ✅")

          # * Resize with the given values
          Path_file = os.path.join(self.Folder, File)
          Image = io.imread(Path_file, as_gray = True)

          #Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)
          #CLAHE = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
          #CLAHE_Imagen = CLAHE.apply(Imagen)

          CLAHE_image = equalize_adapthist(Image, clip_limit = self.Cliplimit)

          Image = img_as_ubyte(Image)
          CLAHE_image = img_as_ubyte(CLAHE_image)

          # * Save each statistic in a variable
          Mae = mae(Image, CLAHE_image)
          Mse = mse(Image, CLAHE_image)
          Ssim = ssim(Image, CLAHE_image)
          Psnr = psnr(Image, CLAHE_image)
          Nrmse = nrmse(Image, CLAHE_image)
          Nmi = nmi(Image, CLAHE_image)
          R2s = r2s(Image, CLAHE_image)

          # * Add the value in the lists already created
          Mae_ALL.append(Mae)
          Mse_ALL.append(Mse)
          Ssim_ALL.append(Ssim)
          Psnr_ALL.append(Psnr)
          Nrmse_ALL.append(Nrmse)
          Nmi_ALL.append(Nmi)
          R2s_ALL.append(R2s)

          # * Name the new file
          Filename_and_technique = Filename + '_CLAHE'
          New_name_filename = Filename_and_technique + Format
          New_folder = os.path.join(self.Newfolder, New_name_filename)

          # * Save the image in a new folder
          io.imsave(New_folder, CLAHE_image)

          # * Save the values of labels and each filenames
          #Images.append(Normalization_Imagen)
          Labels.append(self.Label)
          All_filenames.append(Filename_and_technique)

          Count += 1

      # * Return the new dataframe with the new data
      DataFrame = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

      return DataFrame

  # ? Histogram equalization technique method

  def histogram_equalization_technique(self):

      """
      Get the values from histogram equalization images and save them into a dataframe.

      Parameters:
      argument1 (Folder): Folder chosen.
      argument2 (Folder): Folder destination for new images.
      argument3 (Str): Severity of each image.
      argument4 (Str): Label for each image.

      Returns:
      int:Returning dataframe with all data.
      
      """
      # * Remove all the files in the new folder using this function
      remove_all_files(self.Newfolder)
      
      # * Lists to save the values of the labels and the filename and later use them for a dataframe
      #Images = [] 
      Labels = []
      All_filenames = []

      # * Lists to save the values of each statistic
      Mae_ALL = [] # ? Mean Absolute Error.
      Mse_ALL = [] # ? Mean Squared Error.
      Ssim_ALL = [] # ? Structural similarity.
      Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
      Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
      Nmi_ALL = [] # ? Normalized Mutual Information.
      R2s_ALL = [] # ? Coefficient of determination.

      os.chdir(self.Folder)

      # * Using sort function
      Sorted_files, Total_images = sort_images(self.Folder)
      Count = 1

      # * Reading the files
      for File in Sorted_files:
        
        # * Extract the file name and format
        Filename, Format  = os.path.splitext(File)

        # * Read extension files
        if File.endswith(Format):
          
          print(f"Working with {Count} of {Total_images} {self.Severity} images ✅")
          print(f"Working with {Filename} ✅")

          # * Resize with the given values
          Path_file = os.path.join(self.Folder, File)
          Image = io.imread(Path_file, as_gray = True)

          HE_image = equalize_hist(Image)

          Image = img_as_ubyte(Image)
          HE_image = img_as_ubyte(HE_image)

          # * Save each statistic in a variable
          Mae = mae(Image, HE_image)
          Mse = mse(Image, HE_image)
          Ssim = ssim(Image, HE_image)
          Psnr = psnr(Image, HE_image)
          Nrmse = nrmse(Image, HE_image)
          Nmi = nmi(Image, HE_image)
          R2s = r2s(Image, HE_image)

          # * Add the value in the lists already created
          Mae_ALL.append(Mae)
          Mse_ALL.append(Mse)
          Ssim_ALL.append(Ssim)
          Psnr_ALL.append(Psnr)
          Nrmse_ALL.append(Nrmse)
          Nmi_ALL.append(Nmi)
          R2s_ALL.append(R2s)

          # * Name the new file
          Filename_and_technique = Filename + '_HE'
          New_name_filename = Filename_and_technique + Format
          New_folder = os.path.join(self.Newfolder, New_name_filename)

          # * Save the image in a new folder
          io.imsave(New_folder, HE_image)

          # * Save the values of labels and each filenames
          #Images.append(Normalization_Imagen)
          Labels.append(self.Label)
          All_filenames.append(Filename_and_technique)
          Count += 1

      # * Return the new dataframe with the new data
      DataFrame = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

      return DataFrame

  # ? Unsharp masking technique method

  def unsharp_masking_technique(self):

      """
      Get the values from unsharp masking images and save them into a dataframe.

      Parameters:
      argument1 (Folder): Folder chosen.
      argument2 (Folder): Folder destination for new images.
      argument3 (str): Severity of each image.
      argument4 (float): Radius value use to change Unsharp mask images.
      argument5 (float): Amount value use to change Unsharp mask images.
      argument6 (str): Label for each image.

      Returns:
      int:Returning dataframe with all data.
      
      """
      # * Remove all the files in the new folder using this function
      remove_all_files(self.Newfolder)
      
      # * Lists to save the values of the labels and the filename and later use them for a dataframe
      #Images = [] 
      Labels = []
      All_filenames = []

      # * Lists to save the values of each statistic
      Mae_ALL = [] # ? Mean Absolute Error.
      Mse_ALL = [] # ? Mean Squared Error.
      Ssim_ALL = [] # ? Structural similarity.
      Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
      Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
      Nmi_ALL = [] # ? Normalized Mutual Information.
      R2s_ALL = [] # ? Coefficient of determination.

      os.chdir(self.Folder)

      # * Using sort function
      Sorted_files, Total_images = sort_images(self.Folder)
      Count = 1

      # * Reading the files
      for File in Sorted_files:
        
        # * Extract the file name and format
        Filename, Format  = os.path.splitext(File)

        # * Read extension files
        if File.endswith(Format):
          
          print(f"Working with {Count} of {Total_images} {self.Severity} images ✅")
          print(f"Working with {Filename} ✅")

          # * Resize with the given values
          Path_file = os.path.join(self.Folder, File)
          Image = io.imread(Path_file, as_gray = True)

          UM_image = unsharp_mask(Image, radius = self.Radius, amount = self.Amount)

          Image = img_as_ubyte(Image)
          UM_image = img_as_ubyte(UM_image)

          # * Save each statistic in a variable
          Mae = mae(Image, UM_image)
          Mse = mse(Image, UM_image)
          Ssim = ssim(Image, UM_image)
          Psnr = psnr(Image, UM_image)
          Nrmse = nrmse(Image, UM_image)
          Nmi = nmi(Image, UM_image)
          R2s = r2s(Image, UM_image)

          # * Add the value in the lists already created
          Mae_ALL.append(Mae)
          Mse_ALL.append(Mse)
          Ssim_ALL.append(Ssim)
          Psnr_ALL.append(Psnr)
          Nrmse_ALL.append(Nrmse)
          Nmi_ALL.append(Nmi)
          R2s_ALL.append(R2s)

          # * Name the new file
          Filename_and_technique = Filename + '_UM'
          New_name_filename = Filename_and_technique + Format
          New_folder = os.path.join(self.Newfolder, New_name_filename)

          # * Save the image in a new folder
          io.imsave(New_folder, UM_image)

          # * Save the values of labels and each filenames
          #Images.append(Normalization_Imagen)
          Labels.append(self.Label)
          All_filenames.append(Filename_and_technique)
          Count += 1

      # * Return the new dataframe with the new data
      Dataframe = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

      return Dataframe

  # ? Contrast Stretching technique method

  def contrast_stretching_technique(self):

      """
      Get the values from constrast streching images and save them into a dataframe.

      Parameters:
      argument1 (Folder): Folder chosen.
      argument2 (Folder): Folder destination for new images.
      argument3 (str): Severity of each image.
      argument6 (str): Label for each image.

      Returns:
      int:Returning dataframe with all data.
      
      """
      # * Remove all the files in the new folder using this function
      remove_all_files(self.Newfolder)
      
      # * Lists to save the values of the labels and the filename and later use them for a dataframe
      #Images = [] 
      Labels = []
      All_filenames = []

      # * Lists to save the values of each statistic
      Mae_ALL = [] # ? Mean Absolute Error.
      Mse_ALL = [] # ? Mean Squared Error.
      Ssim_ALL = [] # ? Structural similarity.
      Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
      Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
      Nmi_ALL = [] # ? Normalized Mutual Information.
      R2s_ALL = [] # ? Coefficient of determination.

      os.chdir(self.Folder)

      # * Using sort function
      Sorted_files, Total_images = sort_images(self.Folder)
      Count = 1

      # * Reading the files
      for File in Sorted_files:
        
        # * Extract the file name and format
        Filename, Format  = os.path.splitext(File)

        # * Read extension files
        if File.endswith(Format):
          
          print(f"Working with {Count} of {Total_images} {self.Severity} images ✅")
          print(f"Working with {Filename} ✅")

          # * Resize with the given values
          Path_file = os.path.join(self.Folder, File)
          Image = io.imread(Path_file, as_gray = True)

          p2, p98 = np.percentile(Image, (2, 98))
          CS_image = rescale_intensity(Image, in_range = (p2, p98))

          Image = img_as_ubyte(Image)
          CS_image = img_as_ubyte(CS_image)

          # * Save each statistic in a variable
          Mae = mae(Image, CS_image)
          Mse = mse(Image, CS_image)
          Ssim = ssim(Image, CS_image)
          Psnr = psnr(Image, CS_image)
          Nrmse = nrmse(Image, CS_image)
          Nmi = nmi(Image, CS_image)
          R2s = r2s(Image, CS_image)

          # * Add the value in the lists already created
          Mae_ALL.append(Mae)
          Mse_ALL.append(Mse)
          Ssim_ALL.append(Ssim)
          Psnr_ALL.append(Psnr)
          Nrmse_ALL.append(Nrmse)
          Nmi_ALL.append(Nmi)
          R2s_ALL.append(R2s)

          # * Name the new file
          Filename_and_technique = Filename + '_CS'
          New_name_filename = Filename_and_technique + Format
          New_folder = os.path.join(self.Newfolder, New_name_filename)

          # * Save the image in a new folder
          io.imsave(New_folder, CS_image)

          # * Save the values of labels and each filenames
          #Images.append(Normalization_Imagen)
          Labels.append(self.Label)
          All_filenames.append(Filename_and_technique)
          Count += 1

      # * Return the new dataframe with the new data
      Dataframe = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

      return Dataframe