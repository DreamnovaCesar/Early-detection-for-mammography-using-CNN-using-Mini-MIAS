import os
import cv2
import random
import albumentations as A

from skimage import io

# Data Augmentation

#self, PathFolder, NewPathFolder, Severity, Sampling, Label

class dataAugmentation:
  
  def __init__(self, **kwargs):

    self.folder = kwargs.get('folder')
    self.severity = kwargs.get('severity')
    self.sampling = kwargs.get('sampling')
    self.label = kwargs.get('label')
    self.nfsave = kwargs.get('nfsave', False)

  def ShiftRotation(self, Image_Cropped):

    """
	  Shift rotation using albumentation.

    Parameters:
    argument1 (int): Image chosen.

    Returns:
	  int:Returning image with shift rotation applied
    
   	"""

    transform = A.Compose([
          A.ShiftScaleRotate(p = 1)
      ])
    transformed = transform(image = Image_Cropped)
    Imagen_transformada = transformed["image"]

    return Imagen_transformada

  def FlipHorizontal(self, Image_Cropped):

    """
	  Horizontal flip using albumentation.

    Parameters:
    argument1 (int): Image chosen.

    Returns:
	  int:Returning image with horizontal flip applied
    
   	"""
    transform = A.Compose([
        A.HorizontalFlip(p = 1)
      ])
    transformed = transform(image = Image_Cropped)
    Imagen_transformada = transformed["image"]

    return Imagen_transformada

  def FlipVertical(self, Image_Cropped):

    """
	  Vertical flip using albumentation.

    Parameters:
    argument1 (int): Image chosen.

    Returns:
	  int:Returning image with vertical flip applied
    
   	"""

    transform = A.Compose([
          A.VerticalFlip(p = 1)
        ])
    transformed = transform(image = Image_Cropped)
    Imagen_transformada = transformed["image"]

    return Imagen_transformada

  def Rotation(self, Rotation, Image_Cropped):

    """
	  Rotation using albumentation.

    Parameters:
    argument1 (float): Degrees of rotation.
    argument2 (int): Image chosen.

    Returns:
	  int:Returning image with rotation applied
    
   	"""

    transform = A.Compose([
        A.Rotate(Rotation, p = 1)
      ])
    transformed = transform(image = Image_Cropped)
    Imagen_transformada = transformed["image"]

    return Imagen_transformada

  def DataAugmentation(self):

    """
	  Applying data augmentation different transformations.

    Parameters:
    argument1 (folder): Folder chosen.
    argument2 (str): Severity of each image.
    argument3 (int): Amount of transformation applied for each image, using only rotation.
    argument4 (str): Label for each image.

    Returns:
	  list:Returning images like 'X' value
    list:Returning labels like 'Y' value
    
   	"""
    # Creating variables.

    Existdir = os.path.isdir(self.folder + 'DA') 

    if self.nfsave == True:
      if Existdir == False:
        self.newfolder = self.folder + 'DA'
        os.mkdir(self.newfolder)
      else:
        self.newfolder = self.folder + 'DA'

    Images = [] 
    Labels = [] 

    Rotation_initial_value = -120
    Sampling = 24
    png = ".png"

    # Reading the folder that is used.

    os.chdir(self.folder)
    count = 0

    # The number of images inside the folder.

    images = len(os.listdir(self.folder))

    # Iteration of each image in the folder.

    for File in os.listdir():

      filename, extension  = os.path.splitext(File)

      if File.endswith(png): # Read png files

        print(f"Working with {count} of {images} images of {self.severity}")
        count += 1

        Path_File = os.path.join(self.folder, File)
        Resize_Imagen = cv2.imread(Path_File)
        #Resize_Imagen = cv2.cvtColor(Resize_Imagen, cv2.COLOR_BGR2GRAY)
        #Resize_Imagen = cv2.resize(Resize_Imagen, dim, interpolation = cv2.INTER_CUBIC)

        # 1) Raw

        Images.append(Resize_Imagen)
        Labels.append(self.label)

        if self.nfsave == True:
          
          FilenamesREFNUM = filename  + '_Normal'
          dst = FilenamesREFNUM + png

          dstPath = os.path.join(self.newfolder, dst)
          io.imsave(dstPath, Resize_Imagen)

        # 1.a) Rotation

        for i in range(Sampling):

          Imagen_transformed = self.Rotation(Rotation_initial_value, Resize_Imagen)

          Rotation_initial_value += 10

          Images.append(Imagen_transformed)
          Labels.append(self.label)

          if self.nfsave == True:

            FilenamesREFNUM = filename + '_' + str(i) + '_Rotation' + '_Augmentation'
            dst = FilenamesREFNUM + png

            dstPath = os.path.join(self.newfolder, dst)
            io.imsave(dstPath, Imagen_transformed)

        # 1.b) Flip Vertical

        vertical_transformed = self.FlipVertical(Resize_Imagen)

        Images.append(vertical_transformed)
        Labels.append(self.label)

        if self.nfsave == True:

          FilenamesREFNUM = filename  + '_FlipVertical' + '_Augmentation'
          dst = FilenamesREFNUM + png

          dstPath = os.path.join(self.newfolder, dst)
          io.imsave(dstPath, vertical_transformed)

        for i in range(Sampling):

          Imagen_transformed = self.Rotation(Rotation_initial_value, vertical_transformed)

          Rotation_initial_value += 10

          Images.append(Imagen_transformed)
          Labels.append(self.label)

          if self.nfsave == True:

            FilenamesREFNUM = filename + '_' + str(i) + '_Rotation' + '_FlipVertical' + '_Augmentation'
            dst = FilenamesREFNUM + png

            dstPath = os.path.join(self.newfolder, dst)
            io.imsave(dstPath, Imagen_transformed)

        # 1.c) Flip Horizontal 

        Imagen_transformed = self.FlipHorizontal(Resize_Imagen)

        Images.append(Imagen_transformed)
        Labels.append(self.label)
      
        print(len(Labels))

        if self.nfsave == True:

          FilenamesREFNUM = filename  + '_FlipHorizontal' + '_Augmentation'
          dst = FilenamesREFNUM + png

          dstPath = os.path.join(self.newfolder, dst)
          io.imsave(dstPath, Imagen_transformed)
        
        
    return Images, Labels