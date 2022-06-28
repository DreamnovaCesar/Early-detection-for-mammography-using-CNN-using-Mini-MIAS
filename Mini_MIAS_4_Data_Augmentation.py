import os
import cv2
import random
import albumentations as A

from skimage import io

# Data Augmentation function

class dataAugmentation:
  
  def __init__(self, **kwargs):
    
    # * General parameters
    self.Folder = kwargs.get('Folder')
    self.Severity = kwargs.get('Severity')
    self.Sampling = kwargs.get('Sampling')
    self.Label = kwargs.get('Label')
    self.Saveimages = kwargs.get('Saveimages', False)

    if self.Folder == None:
      raise ValueError("Folder does not exist") #! Alert

    elif self.Severity == None:
      raise ValueError("Add the severity") #! Alert

    elif self.Sampling == None:
      raise ValueError("Add required sampling") #! Alert

    elif self.Label == None:
      raise ValueError("Add the labeling") #! Alert

  # ? shift rotation using albumentation library

  def shift_rotation(self, Image_cropped):

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
    transformed = transform(image = Image_cropped)
    Imagen_transformada = transformed["image"]

    return Imagen_transformada

  # ? Flip horizontal using albumentation library

  def flip_horizontal(self, Image_cropped):

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
    transformed = transform(image = Image_cropped)
    Imagen_transformada = transformed["image"]

    return Imagen_transformada

  # ? Flip vertical using albumentation library

  def flip_vertical(self, Image_cropped):

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
    transformed = transform(image = Image_cropped)
    Imagen_transformada = transformed["image"]

    return Imagen_transformada

  # ? Rotation using albumentation library

  def rotation(self, Rotation, Image_cropped):

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
    transformed = transform(image = Image_cropped)
    Imagen_transformada = transformed["image"]

    return Imagen_transformada

  # ? shift rotation using albumentation library

  def data_augmentation(self):

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
    # * Create a folder with each image and its transformations.
    Exist_dir = os.path.isdir(self.Folder + 'DA') 

    if self.Saveimages == True:
      if Exist_dir == False:
        self.newfolder = self.Folder + 'DA'
        os.mkdir(self.newfolder)
      else:
        self.newfolder = self.Folder + 'DA'

    # * Lists to save the images and their respective labels
    Images = [] 
    Labels = [] 

    # * Initial value to rotate (More information on the albumentation's web)
    Rotation_initial_value = -120
    #Sampling = 24
    #png = ".png"

    # * Reading the folder
    os.chdir(self.Folder)
    Count = 1

    # * The number of images inside the folder
    Total_images = len(os.listdir(self.Folder))

    # * Iteration of each image in the folder.
    for File in os.listdir():

      Filename, Format  = os.path.splitext(File)

      # * Read extension files
      if File.endswith(Format):

        print(f"Working with {Count} of {Total_images} images of {self.Severity}")
        Count += 1

        # * Resize with the given values
        Path_file = os.path.join(self.Folder, File)
        Image = cv2.imread(Path_file)
        #Resize_Imagen = cv2.cvtColor(Resize_Imagen, cv2.COLOR_BGR2GRAY)
        #Resize_Imagen = cv2.resize(Resize_Imagen, dim, interpolation = cv2.INTER_CUBIC)

        # ? 1) Standard image

        Images.append(Image)
        Labels.append(self.Label)

        # * if this parameter is true all images will be saved in a new folder
        if self.Saveimages == True:
          
          Filename_and_label = Filename + '_Normal'
          New_name_filename = Filename_and_label + Format
          New_folder = os.path.join(self.newfolder, New_name_filename)

          io.imsave(New_folder, Image)

        # ? 1.A) Flip horizontal 

        Image_flip_horizontal = self.flip_horizontal(Image)

        Images.append(Image_flip_horizontal)
        Labels.append(self.Label)

        # * if this parameter is true all images will be saved in a new folder
        if self.Saveimages == True:

          Filename_and_label = Filename + '_FlipHorizontal' + '_Augmentation'
          New_name_filename = Filename_and_label + Format
          New_folder = os.path.join(self.newfolder, New_name_filename)

          io.imsave(New_folder, Image_flip_horizontal)

        # ? 1.B) Rotation

        # * this 'for' increments the rotation angles of each image
        for i in range(self.Sampling):

          Image_rotation = self.rotation(Rotation_initial_value, Image)

          Rotation_initial_value += 10

          Images.append(Image_rotation)
          Labels.append(self.Label)

          # * if this parameter is true all images will be saved in a new folder
          if self.Saveimages == True:

            Filename_and_label = Filename + '_' + str(i) + '_Rotation' + '_Augmentation'
            New_name_filename = Filename_and_label + Format
            New_folder = os.path.join(self.newfolder, New_name_filename)

            io.imsave(New_folder, Image_rotation)

        # ? 2.A) Flip vertical

        Image_flip_vertical = self.flip_vertical(Image)

        Images.append(Image_flip_vertical)
        Labels.append(self.Label)

        # * if this parameter is true all images will be saved in a new folder
        if self.Saveimages == True:

          Filename_and_label = Filename + '_FlipVertical' + '_Augmentation'
          New_name_filename = Filename_and_label + Format
          New_folder = os.path.join(self.newfolder, New_name_filename)

          io.imsave(New_folder, Image_flip_vertical)
        
        # ? 2.B) Rotation

        # * this 'for' increments the rotation angles of each image
        for i in range(self.Sampling):

          Image_flip_vertical_rotation = self.rotation(Rotation_initial_value, Image_flip_vertical)

          Rotation_initial_value += 10

          Images.append(Image_flip_vertical_rotation)
          Labels.append(self.Label)

          # * if this parameter is true all images will be saved in a new folder
          if self.Saveimages == True:

            Filename_and_label = Filename + '_' + str(i) + '_Rotation' + '_FlipVertical' + '_Augmentation'
            New_name_filename = Filename_and_label + Format
            New_folder = os.path.join(self.newfolder, New_name_filename)

            io.imsave(New_folder, Image_flip_vertical_rotation)
        
    return Images, Labels