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

    self.folder = kwargs.get('folder', None)
    self.newfolder = kwargs.get('newfolder', None)
    self.severity = kwargs.get('severity', None)
    self.label = kwargs.get('label', None)

    self.interpolation = kwargs.get('interpolation', None)
    self.Xresize = kwargs.get('Xresize', None)
    self.Yresize = kwargs.get('Yresize', None)

    self.division = kwargs.get('division', 3)

    self.cliplimit = kwargs.get('cliplimit', 0.01)

    self.radius = kwargs.get('radius', 1)
    self.amount = kwargs.get('amount', 1)

    if self.folder == None:
      raise ValueError("Folder does not exist")

    if self.folder == None:
      raise ValueError("New folder destination does not exist")

    if self.folder == None:
      raise ValueError("Assign the severity")

    if self.folder == None:
      raise ValueError("Assign the interpolation that will be used")

  # Resize technique.

  def resize_technique(self):

    Images = [] 

    os.chdir(self.folder)

    print(os.getcwd())
    print("\n")

    sorted_files, images = sort_images(self.folder)
    count = 1

    for File in sorted_files:

      filename, extension  = os.path.splitext(File)

      if File.endswith(extension): # Read extension files

        try:
          print(f"Working with {count} of {images} normal images")
          count += 1

          Path_File = os.path.join(self.folder, File)
          Imagen = cv2.imread(Path_File)

          dsize = (self.Xresize, self.Yresize)
          Resized_Imagen = cv2.resize(Imagen, dsize, interpolation = self.interpolation)

          print(Imagen.shape, ' -------- ', Resized_Imagen.shape)

          dst = filename + extension
          dstPath_N = os.path.join(self.folder, dst)

          cv2.imwrite(dstPath_N, Resized_Imagen)
          Images.append(Resized_Imagen)
          
        except OSError:
          print('Cannot convert %s ❌' % File)

    print("\n")
    print(f"COMPLETE {count} of {images} RESIZED ✅")

  # Normalization technique.

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
    remove_all_files(self.newfolder)

    #Images = [] 
    Labels = []
    Filename_ALL = []

    Mae_ALL = [] # MSE normal.
    Mse_ALL = [] # PSNR normal.
    Ssim_ALL = [] # MSE normal.
    Psnr_ALL = [] # PSNR normal.
    Nrmse_ALL = [] # MSE normal.
    Nmi_ALL = [] # PSNR normal.
    R2s_ALL = [] # PSNR normal.

    # Normals Images.

    os.chdir(self.folder)

    sorted_files, images = sort_images(self.folder)
    Count = 1

    for File in sorted_files:

      filename, extension  = os.path.splitext(File)

      if File.endswith(extension): # Read extension files
        
        print(f"Working with {Count} of {images} {self.severity} images ✅")
        print(f"Working with {filename} ✅")

        Path_File = os.path.join(self.folder, File)
        Imagen = cv2.imread(Path_File)

        Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)

        #print("%s has %d and %d" % (File, Imagen.shape[0], Imagen.shape[1]))

        Norm_img = np.zeros((Imagen.shape[0], Imagen.shape[1]))
        Normalization_Imagen = cv2.normalize(Imagen, Norm_img, 0, 255, cv2.NORM_MINMAX)

        Mae = mae(Imagen, Normalization_Imagen)
        Mse = mse(Imagen, Normalization_Imagen)
        Ssim = ssim(Imagen, Normalization_Imagen)
        Psnr = psnr(Imagen, Normalization_Imagen)
        Nrmse = nrmse(Imagen, Normalization_Imagen)
        Nmi = nmi(Imagen, Normalization_Imagen)
        R2s = r2s(Imagen, Normalization_Imagen)

        Mae_ALL.append(Mae)
        Mse_ALL.append(Mse)
        Ssim_ALL.append(Ssim)
        Psnr_ALL.append(Psnr)
        Nrmse_ALL.append(Nrmse)
        Nmi_ALL.append(Nmi)
        R2s_ALL.append(R2s)

        FilenamesREFNUM = filename + '_Normalization'
        dst = FilenamesREFNUM + extension
        dstPath = os.path.join(self.newfolder, dst)
        
        #Normalization_Imagen = Normalization_Imagen.astype('float32')
        #Normalization_Imagen = Normalization_Imagen / 255.0
        #print(Normalization_Imagen)

        cv2.imwrite(dstPath, Normalization_Imagen)

        REFNUM = FilenamesREFNUM

        #Images.append(Normalization_Imagen)
        Labels.append(self.label)
        Filename_ALL.append(REFNUM)

        Count += 1

    Dataframe = pd.DataFrame({'REFNUMMF_ALL':Filename_ALL, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

    return Dataframe

  # Median filter technique.

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
      # Remove images of the folder chosen

      remove_all_files(self.newfolder)
      
      # General lists

      #Images = [] 
      Labels = []
      Filename_ALL = []

      # Metric lists

      Mae_ALL = [] 
      Mse_ALL = [] 
      Ssim_ALL = [] 
      Psnr_ALL = [] 
      Nrmse_ALL = [] 
      Nmi_ALL = [] 
      R2s_ALL = [] 

      # Extension used 

      os.chdir(self.folder)

      sorted_files, images = sort_images(self.folder)
      Count = 1

      # For each file sorted.

      for File in sorted_files:

        filename, extension  = os.path.splitext(File)

        if File.endswith(extension): # Read extension files
          
          # Using median filter function

          print(f"Working with {Count} of {images} {self.severity} images ✅")
          print(f"Working with {filename} ✅")

          Path_File = os.path.join(self.folder, File)
          Imagen = io.imread(Path_File, as_gray = True)

          #Image_Median_Filter = cv2.medianBlur(Imagen, Division)
          MedianFilter_Image = filters.median(Imagen, np.ones((self.division, self.division)))

          Mae = mae(Imagen, MedianFilter_Image)
          Mse = mse(Imagen, MedianFilter_Image)
          Ssim = ssim(Imagen, MedianFilter_Image)
          Psnr = psnr(Imagen, MedianFilter_Image)
          Nrmse = nrmse(Imagen, MedianFilter_Image)
          Nmi = nmi(Imagen, MedianFilter_Image)
          R2s = r2s(Imagen, MedianFilter_Image)

          Mae_ALL.append(Mae)
          Mse_ALL.append(Mse)
          Ssim_ALL.append(Ssim)
          Psnr_ALL.append(Psnr)
          Nrmse_ALL.append(Nrmse)
          Nmi_ALL.append(Nmi)
          R2s_ALL.append(R2s)

          FilenamesREFNUM = filename + '_Median_Filter'
          dst = FilenamesREFNUM + extension
          dstPath = os.path.join(self.newfolder, dst)

          io.imsave(dstPath, MedianFilter_Image)

          REFNUM = FilenamesREFNUM

          Count += 1

          # Saving values in the lists

          #Images.append(MedianFilter_Image)
          Labels.append(self.label)
          Filename_ALL.append(REFNUM)

      # Final dataframe

      DataFrame = pd.DataFrame({'REFNUMMF_ALL':Filename_ALL, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

      return DataFrame

  # CLAHE technique.

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

      # Remove images of the folder chosen

      remove_all_files(self.newfolder)
      
      # General lists

      #Images = [] 
      Labels = []
      Filename_ALL = []

      # Metrics lists

      Mae_ALL = [] 
      Mse_ALL = [] 
      Ssim_ALL = [] 
      Psnr_ALL = [] 
      Nrmse_ALL = [] 
      Nmi_ALL = [] 
      R2s_ALL = [] 

      # Extension used

      os.chdir(self.folder)

      sorted_files, images = sort_images(self.folder)
      Count = 1

      # For each file sorted.

      for File in sorted_files:

        filename, extension  = os.path.splitext(File)

        if File.endswith(extension): # Read extension files
          
          # Using CLAHE function

          print(f"Working with {Count} of {images} {self.severity} images ✅")
          print(f"Working with {filename} ✅")

          Path_File = os.path.join(self.folder, File)
          Imagen = io.imread(Path_File, as_gray = True)

          #Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)
          #CLAHE = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
          #CLAHE_Imagen = CLAHE.apply(Imagen)

          CLAHE_Imagen = equalize_adapthist(Imagen, clip_limit = self.cliplimit)

          Imagen = img_as_ubyte(Imagen)
          CLAHE_Imagen = img_as_ubyte(CLAHE_Imagen)

          Mae = mae(Imagen, CLAHE_Imagen)
          Mse = mse(Imagen, CLAHE_Imagen)
          Ssim = ssim(Imagen, CLAHE_Imagen)
          Psnr = psnr(Imagen, CLAHE_Imagen)
          Nrmse = nrmse(Imagen, CLAHE_Imagen)
          Nmi = nmi(Imagen, CLAHE_Imagen)
          R2s = r2s(Imagen, CLAHE_Imagen)

          Mae_ALL.append(Mae)
          Mse_ALL.append(Mse)
          Ssim_ALL.append(Ssim)
          Psnr_ALL.append(Psnr)
          Nrmse_ALL.append(Nrmse)
          Nmi_ALL.append(Nmi)
          R2s_ALL.append(R2s)

          FilenamesREFNUM = filename + '_CLAHE'
          print(FilenamesREFNUM)

          dst = str(FilenamesREFNUM) + extension
          dstPath = os.path.join(self.newfolder, dst)

          io.imsave(dstPath, CLAHE_Imagen)

          REFNUM = FilenamesREFNUM

          Count += 1

          # Saving values in the lists

          #Images.append(CLAHE_Imagen)
          Labels.append(self.label)
          Filename_ALL.append(REFNUM)

      # Final dataframe

      DataFrame = pd.DataFrame({'REFNUMMF_ALL':Filename_ALL, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

      return DataFrame

  # Histogram equalization technique.

  def HistogramEqualization(self):

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

      # Remove images of the folder chosen

      remove_all_files(self.newfolder)
      
      # General lists

      #Images = [] 
      Labels = []
      Filename_ALL = []

      # Statistic lists

      Mae_ALL = [] 
      Mse_ALL = [] 
      Ssim_ALL = [] 
      Psnr_ALL = [] 
      Nrmse_ALL = [] 
      Nmi_ALL = [] 
      R2s_ALL = [] 

      # Extension used

      os.chdir(self.folder)

      sorted_files, images = sort_images(self.folder)
      Count = 1

      # For each file sorted.

      for File in sorted_files:

        filename, extension  = os.path.splitext(File)

        if File.endswith(extension): # Read extension files

          # Using CLAHE function

          print(f"Working with {Count} of {images} {self.severity} images ✅")
          print(f"Working with {filename} ✅")

          Path_File = os.path.join(self.folder, File)
          Imagen = io.imread(Path_File, as_gray = True)

          HE_Imagen = equalize_hist(Imagen)

          Imagen = img_as_ubyte(Imagen)
          HE_Imagen = img_as_ubyte(HE_Imagen)

          Mae = mae(Imagen, HE_Imagen)
          Mse = mse(Imagen, HE_Imagen)
          Ssim = ssim(Imagen, HE_Imagen)
          Psnr = psnr(Imagen, HE_Imagen)
          Nrmse = nrmse(Imagen, HE_Imagen)
          Nmi = nmi(Imagen, HE_Imagen)
          R2s = r2s(Imagen, HE_Imagen)

          Mae_ALL.append(Mae)
          Mse_ALL.append(Mse)
          Ssim_ALL.append(Ssim)
          Psnr_ALL.append(Psnr)
          Nrmse_ALL.append(Nrmse)
          Nmi_ALL.append(Nmi)
          R2s_ALL.append(R2s)

          FilenamesREFNUM = filename + '_HE'
          dst = FilenamesREFNUM + extension
          dstPath = os.path.join(self.newfolder, dst)

          io.imsave(dstPath, img_as_ubyte(HE_Imagen))

          REFNUM = FilenamesREFNUM

          Count += 1

          # Saving values in the lists

          #Images.append(HE_Imagen)
          Labels.append(self.label)
          Filename_ALL.append(REFNUM)

      # Final dataframe

      DataFrame = pd.DataFrame({'REFNUMMF_ALL':Filename_ALL, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

      return DataFrame

  # Unsharp masking technique.

  def UnsharpMasking(self):

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

      # Remove images of the folder chosen

      remove_all_files(self.newfolder)
      
      # General lists

      #Images = [] 
      Labels = []
      Filename_ALL = []

      # Metrics lists

      Mae_ALL = [] 
      Mse_ALL = [] 
      Ssim_ALL = [] 
      Psnr_ALL = [] 
      Nrmse_ALL = [] 
      Nmi_ALL = [] 
      R2s_ALL = [] 

      # Extension used  

      os.chdir(self.folder)

      sorted_files, images = sort_images(self.folder)
      Count = 1

      # For each file sorted.

      for File in sorted_files:

        filename, extension = os.path.splitext(File)

        if File.endswith(extension): # Read extension files
          
          # Using unsharp masking function

          print(f"Working with {Count} of {images} {self.severity} images ✅")
          print(f"Working with {filename} ✅")

          Path_File = os.path.join(self.folder, File)
          Imagen = io.imread(Path_File, as_gray = True)

          UM_Imagen = unsharp_mask(Imagen, radius = self.radius, amount = self.amount)

          Imagen = img_as_ubyte(Imagen)
          UM_Imagen = img_as_ubyte(UM_Imagen)

          Mae = mae(Imagen, UM_Imagen)
          Mse = mse(Imagen, UM_Imagen)
          Ssim = ssim(Imagen, UM_Imagen)
          Psnr = psnr(Imagen, UM_Imagen)
          Nrmse = nrmse(Imagen, UM_Imagen)
          Nmi = nmi(Imagen, UM_Imagen)
          R2s = r2s(Imagen, UM_Imagen)

          Mae_ALL.append(Mae)
          Mse_ALL.append(Mse)
          Ssim_ALL.append(Ssim)
          Psnr_ALL.append(Psnr)
          Nrmse_ALL.append(Nrmse)
          Nmi_ALL.append(Nmi)
          R2s_ALL.append(R2s)

          FilenamesREFNUM = filename + '_UM'
          dst = FilenamesREFNUM + extension
          dstPath = os.path.join(self.newfolder, dst)

          io.imsave(dstPath, img_as_ubyte(UM_Imagen))

          REFNUM = FilenamesREFNUM

          Count += 1

          # Saving values in the lists

          #Images.append(UM_Imagen)
          Labels.append(self.label)
          Filename_ALL.append(REFNUM)

      # Final dataframe

      Dataframe = pd.DataFrame({'REFNUMMF_ALL':Filename_ALL, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

      return Dataframe

  # Contrast Stretching technique.

  def ContrastStretching(self):

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

      # Remove images of the folder chosen

      remove_all_files(self.newfolder)
      
      # General lists

      #Images = [] 
      Labels = []
      Filename_ALL = []

      # Metrics lists

      Mae_ALL = [] 
      Mse_ALL = [] 
      Ssim_ALL = [] 
      Psnr_ALL = [] 
      Nrmse_ALL = [] 
      Nmi_ALL = [] 
      R2s_ALL = [] 

      # Extension used

      os.chdir(self.folder)

      sorted_files, images = sort_images(self.folder)
      Count = 1

      # For each file sorted.

      for File in sorted_files: # Read extension files

        filename, extension  = os.path.splitext(File)

        if File.endswith(extension): # Read png files
          
          # Using unsharp masking function

          print(f"Working with {Count} of {images} {self.severity} images ✅")
          print(f"Working with {filename} ✅")

          Path_File = os.path.join(self.folder, File)
          Imagen = io.imread(Path_File, as_gray = True)

          p2, p98 = np.percentile(Imagen, (2, 98))
          CS_Imagen = rescale_intensity(Imagen, in_range = (p2, p98))

          Imagen = img_as_ubyte(Imagen)
          CS_Imagen = img_as_ubyte(CS_Imagen)

          Mae = mae(Imagen, CS_Imagen)
          Mse = mse(Imagen, CS_Imagen)
          Ssim = ssim(Imagen, CS_Imagen)
          Psnr = psnr(Imagen, CS_Imagen)
          Nrmse = nrmse(Imagen, CS_Imagen)
          Nmi = nmi(Imagen, CS_Imagen)
          R2s = r2s(Imagen, CS_Imagen)

          Mae_ALL.append(Mae)
          Mse_ALL.append(Mse)
          Ssim_ALL.append(Ssim)
          Psnr_ALL.append(Psnr)
          Nrmse_ALL.append(Nrmse)
          Nmi_ALL.append(Nmi)
          R2s_ALL.append(R2s)

          FilenamesREFNUM = filename + '_CS'
          dst = FilenamesREFNUM + extension
          dstPath = os.path.join(self.newfolder, dst)

          io.imsave(dstPath, img_as_ubyte(CS_Imagen))

          REFNUM = FilenamesREFNUM

          Count += 1

          # Saving values in the lists

          #Images.append(CS_Imagen)
          Labels.append(self.label)
          Filename_ALL.append(REFNUM)

      # Final dataframe

      DataFrame = pd.DataFrame({'REFNUMMF_ALL':Filename_ALL, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

      return DataFrame