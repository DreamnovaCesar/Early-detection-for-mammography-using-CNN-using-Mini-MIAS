import os
import cv2
import numpy as np
import pandas as pd

from glrlm import GLRLM
from skimage.feature import graycomatrix, graycoprops

from Mini_MIAS_2_General_Functions import sort_images

# First Order features from https://github.com/giakou4/pyfeats/blob/main/pyfeats/textural/fos.py

def fos(f, mask):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    Returns
    -------
    features : numpy ndarray
        1)Mean, 2)Variance, 3)Median (50-Percentile), 4)Mode, 
        5)Skewness, 6)Kurtosis, 7)Energy, 8)Entropy, 
        9)Minimal Gray Level, 10)Maximal Gray Level, 
        11)Coefficient of Variation, 12,13,14,15)10,25,75,90-
        Percentile, 16)Histogram width
    labels : list
        Labels of features.
    '''
    if mask is None:
        mask = np.ones(f.shape)
    
    # 1) Labels
    labels = ["FOS_Mean","FOS_Variance","FOS_Median","FOS_Mode","FOS_Skewness",
              "FOS_Kurtosis","FOS_Energy","FOS_Entropy","FOS_MinimalGrayLevel",
              "FOS_MaximalGrayLevel","FOS_CoefficientOfVariation",
              "FOS_10Percentile","FOS_25Percentile","FOS_75Percentile",
              "FOS_90Percentile","FOS_HistogramWidth"]
    
    # 2) Parameters
    f  = f.astype(np.uint8)
    mask = mask.astype(np.uint8)
    level_min = 0
    level_max = 255
    Ng = (level_max - level_min) + 1
    bins = Ng
    
    # 3) Calculate Histogram H inside ROI
    f_ravel = f.ravel() 
    mask_ravel = mask.ravel() 
    roi = f_ravel[mask_ravel.astype(bool)] 
    H = np.histogram(roi, bins = bins, range = [level_min, level_max], density = True)[0]
    
    # 4) Calculate Features
    features = np.zeros(16, np.double)  
    i = np.arange(0, bins)
    features[0] = np.dot(i, H)
    features[1] = sum(np.multiply((( i- features[0]) ** 2), H))
    features[2] = np.percentile(roi, 50) 
    features[3] = np.argmax(H)
    features[4] = sum(np.multiply(((i-features[0]) ** 3), H)) / (np.sqrt(features[1]) ** 3)
    features[5] = sum(np.multiply(((i-features[0]) ** 4), H)) / (np.sqrt(features[1]) ** 4)
    features[6] = sum(np.multiply(H, H))
    features[7] = -sum(np.multiply(H, np.log(H + 1e-16)))
    features[8] = min(roi)
    features[9] = max(roi)
    features[10] = np.sqrt(features[2]) / features[0]
    features[11] = np.percentile(roi, 10) 
    features[12] = np.percentile(roi, 25)  
    features[13] = np.percentile(roi, 75) 
    features[14] = np.percentile(roi, 90) 
    features[15] = features[14] - features[11]
    
    return features, labels

# class for features extraction using first order statistic and GLCM.

class featureExtraction():

  def __init__(self, **kwargs):
    
    self.Folder = kwargs.get('Folder', None)
    #self.newfolder = kwargs.get('newfolder', None)
    self.Format = kwargs.get('Format', None)
    self.Images = kwargs.get('Images', None)
    #self.Images_tumor = kwargs.get('Images_tumor', None)
    self.Label = kwargs.get('Label', None)
    #self.Label_tumor = kwargs.get('Label_tumor', None)

    #if self.Folder == None:
      #raise ValueError("Folder does not exist") #! Alert

    #elif self.Format == None:
      #raise ValueError("Assign the format") #! Alert

    #elif self.Label == None:
      #raise ValueError("Assign the label") #! Alert

  # ? FOF features folder

  def textures_Feature_first_order_from_folder(self):
    
    # * General lists
    #Images = []
    Labels = []
    All_filename = [] 

    # * First order tag
    Fof = 'First Order Features'
    
    # * Lists for the statistics
    Mean = []
    Var = []
    Skew = []
    Kurtosis = []
    Energy = []
    Entropy = []

    os.chdir(self.folder)

    # * Using sort function
    Sorted_files, Total_images = sort_images(self.Folder)
    count = 1

    # * Reading the files
    for File in Sorted_files:
      if File.endswith(self.Format):

        try:

            Filename, Format  = os.path.splitext(File)
            print(f"Working with {count} of {Total_images} {Format} images, {Filename} ------- {self.Format} ✅")
            count += 1

            # * Reading the image
            Path_File = os.path.join(self.Folder, File)
            Image = cv2.imread(Path_File)
            
            # ? mean = np.mean(Imagen)
            # ? std = np.std(Imagen)
            # ? entropy = shannon_entropy(Imagen)
            # ? kurtosis_ = kurtosis(Imagen, axis = None)
            # ? skew_ = skew(Imagen, axis = None)
            # ? labels = ["FOS_Mean","FOS_Variance","FOS_Median","FOS_Mode","FOS_Skewness",
            # ? "FOS_Kurtosis","FOS_Energy","FOS_Entropy","FOS_MinimalGrayLevel",
            # ? "FOS_MaximalGrayLevel","FOS_CoefficientOfVariation",
            # ? "FOS_10Percentile","FOS_25Percentile","FOS_75Percentile",
            # ? "FOS_90Percentile","FOS_HistogramWidth"]
            
            # * Extracting the first order features from the fos function
            Features, Labels_ = fos(Image, None)

            # * Add the value in the lists already created
            Mean.append(Features[0])
            Var.append(Features[1])
            Skew.append(Features[4])
            Kurtosis.append(Features[5])
            Energy.append(Features[6])
            Entropy.append(Features[7])

            Labels.append(self.Label)
            All_filename.append(Filename)
            #Extensions.append(extension)

            #print(len(Mean))
            #print(len(Var))
            #print(len(Skew))
            #print(len(Kurtosis))
            #print(len(Energy))
            #print(len(Entropy))
            #print(len(Labels))
            #print(len(Filename))

        except OSError:
            print('Cannot convert %s ❌' % File)

    # * Return the new dataframe with the new data
    Dataset = pd.DataFrame({'REFNUM':Filename, 'Mean':Mean, 'Var':Var, 'Kurtosis':Kurtosis, 'Energy':Energy, 'Skew':Skew, 'Entropy':Entropy, 'Labels':Labels})

    # * Return a dataframe with only the data without the labels
    X = Dataset.iloc[:, [1, 2, 3, 4, 5, 6]].values

    # * Return a dataframe with only the labels
    Y = Dataset.iloc[:, 0].values

    # * Return the three dataframes
    return Dataset, X, Y

  # ? GLRLM features folder

  def textures_Feature_GLRLM_from_folder(self):
    
    # * General lists
    #Images = []
    Labels = []
    All_filename = [] 

    # * GLRLM tag
    Glrlm = 'Gray-Level Run Length Matrix'

    # * Lists for the statistics
    SRE = []  # Short Run Emphasis
    LRE  = [] # Long Run Emphasis
    GLU = []  # Grey Level Uniformity
    RLU = []  # Run Length Uniformity
    RPC = []  # Run Percentage

    os.chdir(self.folder)

    # * Using sort function
    Sorted_files, Total_images = sort_images(self.Folder)
    count = 1

    # * Reading the file
    for File in Sorted_files:
      if File.endswith(self.Format): 

        try:

            Filename, Format  = os.path.splitext(File)
            print(f"Working with {count} of {Total_images} {Format} images, {Filename} ------- {self.Format} ✅")
            count += 1

            # * Reading the image
            Path_File = os.path.join(self.folder, File)
            Imagen = cv2.imread(Path_File)
            Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)

            app = GLRLM()
            glrlm = app.get_features(Imagen, 8)
            print(glrlm.Features)

            # * Add the value in the lists already created
            SRE.append(glrlm.Features[0])
            LRE.append(glrlm.Features[1])
            GLU.append(glrlm.Features[2])
            RLU.append(glrlm.Features[3])
            RPC.append(glrlm.Features[4])

            Labels.append(self.Label)
            All_filename.append(Filename)

        except OSError:
            print('Cannot convert %s ❌' % File)

    # * Return the new dataframe with the new data
    Dataset = pd.DataFrame({'REFNUM':Filename, 'SRE':SRE, 'LRE':LRE, 'GLU':GLU, 'RLU':RLU, 'RPC':RPC, 'Labels':Labels})

    # * Return a dataframe with only the data without the labels
    X = Dataset.iloc[:, [1, 2, 3, 4, 5]].values

    # * Return a dataframe with only the labels
    Y = Dataset.iloc[:, -1].values

    # * Return the three dataframes
    return Dataset, X, Y

  # ? GLCM features folder

  def textures_Feature_GLCM_from_folder(self):
    
    # * General lists
    #Images = []
    Labels = []
    All_filename = [] 

    # * GLCM tag
    Glcm = 'Gray-Level Co-Occurance Matrix'

    # * Lists for the statistics
    Dissimilarity = []
    Correlation = []
    Homogeneity = []
    Energy = []
    Contrast = []
    ASM = []

    os.chdir(self.folder)

    # * Using sort function
    sorted_files, images = sort_images(self.Folder)
    count = 1

    # * Reading the file
    for File in sorted_files:
      if File.endswith(self.Format): # Read png files

        try:
            Filename, Format  = os.path.splitext(File)
            print(f"Working with {count} of {images} {Format} images, {Filename} ------- {self.Format} ✅")
            count += 1

            # * Reading the image
            Path_File = os.path.join(self.folder, File)
            Imagen = cv2.imread(Path_File)
            Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)

            # * Add the value in the lists already created
            GLCM = graycomatrix(Imagen, [1], [0, np.pi/4, np.pi/2, 3 * np.pi/4])
            Energy.append(graycoprops(GLCM, 'energy')[0, 0])
            Correlation.append(graycoprops(GLCM, 'correlation')[0, 0])
            Homogeneity.append(graycoprops(GLCM, 'homogeneity')[0, 0])
            Dissimilarity.append(graycoprops(GLCM, 'dissimilarity')[0, 0])
            Contrast.append(graycoprops(GLCM, 'contrast')[0, 0])
            ASM.append(graycoprops(GLCM, 'ASM')[0, 0])

            Labels.append(self.Label)
            All_filename.append(Filename)

        except OSError:
            print('Cannot convert %s ❌' % File)

    # * Return the new dataframe with the new data
    Dataset = pd.DataFrame({'REFNUM':Filename, 'Energy':Energy, 'Correlation':Correlation, 'Homogeneity':Homogeneity, 'Dissimilarity':Dissimilarity, 'Contrast':Contrast, 'ASM':ASM, 'Labels':Labels})
    
    # * Return a dataframe with only the data without the labels
    X = Dataset.iloc[:, [1, 2, 3, 4, 5, 6]].values

    # * Return a dataframe with only the labels
    Y = Dataset.iloc[:, 0].values

    # * Return the three dataframes
    return Dataset, X, Y

  # ? FOF features images

  def textures_Feature_first_order_from_images(self):
    
    # * General lists
    Labels = []

    # * First order tag
    Fof = 'First Order Features'

    # * Lists for the statistics
    Mean = []
    Var = []
    Skew = []
    Kurtosis = []
    Energy = []
    Entropy = []

    count = 1

    # * Reading the file
    for File in range(len(self.Images)):

        try:

            print(f"Working with {count} of {len(self.Images)} images ✅")
            count += 1

            # * Reading the image
            self.Images[File] = cv2.cvtColor(self.Images[File], cv2.COLOR_BGR2GRAY)

            # * Extracting the first order features from the fos function
            Features, Labels_ = fos(self.Images[File], None)

            # * Add the value in the lists already created
            Mean.append(Features[0])
            Var.append(Features[1])
            Skew.append(Features[4])
            Kurtosis.append(Features[5])
            Energy.append(Features[6])
            Entropy.append(Features[7])
            Labels.append(self.Label[0])

        except OSError:
            print('Cannot convert %s ❌' % File)

    # * Return the new dataframe with the new data
    Dataset = pd.DataFrame({'Mean':Mean, 'Var':Var, 'Kurtosis':Kurtosis, 'Energy':Energy, 'Skew':Skew, 'Entropy':Entropy, 'Labels':Labels})

    # * Return a dataframe with only the data without the labels
    X = Dataset.iloc[:, [0, 1, 2, 3, 4, 5]].values

    # * Return a dataframe with only the labels
    Y = Dataset.iloc[:, -1].values

    # * Return the three dataframes
    return Dataset, X, Y, Fof

  # ? GLRLM features images

  def textures_Feature_GLRLM_from_images(self):

    # * General lists
    Labels = []

    # * GLRLM tag
    Glrlm = 'Gray-Level Run Length Matrix'

    # * Lists for the statistics
    SRE = []  # Short Run Emphasis
    LRE  = [] # Long Run Emphasis
    GLU = []  # Grey Level Uniformity
    RLU = []  # Run Length Uniformity
    RPC = []  # Run Percentage

    count = 1

    # * Reading the file
    for File in range(len(self.Images)):

        try:
            print(f"Working with {count} of {len(self.Images)} images ✅")
            count += 1

            # * Reading the image
            app = GLRLM()
            glrlm = app.get_features(self.Images[File], 8)

            # * Add the value in the lists already created
            SRE.append(glrlm.Features[0])
            LRE.append(glrlm.Features[1])
            GLU.append(glrlm.Features[2])
            RLU.append(glrlm.Features[3])
            RPC.append(glrlm.Features[4])
            Labels.append(self.Label[0])

        except OSError:
            print('Cannot convert %s ❌' % File)

    # * Return the new dataframe with the new data
    Dataset = pd.DataFrame({'SRE':SRE, 'LRE':LRE, 'GLU':GLU, 'RLU':RLU, 'RPC':RPC, 'Labels':Labels})

    # * Return a dataframe with only the data without the labels
    X = Dataset.iloc[:, [0, 1, 2, 3, 4]].values

    # * Return a dataframe with only the labels
    Y = Dataset.iloc[:, -1].values

    # * Return the three dataframes
    return Dataset, X, Y, Glrlm

  # ? GLCM features images

  def textures_Feature_GLCM_from_images(self):

    # * General lists
    Labels = []

    # * GLCM tag
    Glcm = 'Gray-Level Co-Occurance Matrix'

    # * Create empty dataframe
    DataFrame = pd.DataFrame()
    
    # * Lists for the statistics
    Dissimilarity_1_0 = []
    Correlation_1_0 = []
    Homogeneity_1_0 = []
    Energy_1_0 = []
    Contrast_1_0 = []

    Dissimilarity_1_pi_4 = []
    Correlation_1_pi_4 = []
    Homogeneity_1_pi_4 = []
    Energy_1_pi_4 = []
    Contrast_1_pi_4 = []

    Dissimilarity_7_pi_2 = []
    Correlation_7_pi_2 = []
    Homogeneity_7_pi_2 = []
    Energy_7_pi_2 = []
    Contrast_7_pi_2 = []

    Dissimilarity_7_3_pi_4 = []
    Correlation_7_3_pi_4 = []
    Homogeneity_7_3_pi_4 = []
    Energy_7_3_pi_4 = []
    Contrast_7_3_pi_4 = []

    count = 1

    # * Reading the file
    for File in range(len(self.Images)):

        try:

            print(f"Working with {count} of {len(self.Images)} images ✅")
            count += 1

            # * Reading the image
            self.Images[File] = cv2.cvtColor(self.Images[File], cv2.COLOR_BGR2GRAY)

            # * Add the value in the lists already created
            GLCM_1_0 = graycomatrix(self.Images[File], [1], [0])
            Energy_1_0.append(graycoprops(GLCM_1_0, 'energy')[0, 0])
            Correlation_1_0.append(graycoprops(GLCM_1_0, 'correlation')[0, 0])
            Homogeneity_1_0.append(graycoprops(GLCM_1_0, 'homogeneity')[0, 0])
            Dissimilarity_1_0.append(graycoprops(GLCM_1_0, 'dissimilarity')[0, 0])
            Contrast_1_0.append(graycoprops(GLCM_1_0, 'contrast')[0, 0])
          
            GLCM_1_pi_4 = graycomatrix(self.Images[File], [1], [np.pi/4])
            Energy_1_pi_4.append(graycoprops(GLCM_1_pi_4, 'energy')[0, 0])
            Correlation_1_pi_4.append(graycoprops(GLCM_1_pi_4, 'correlation')[0, 0])
            Homogeneity_1_pi_4.append(graycoprops(GLCM_1_pi_4, 'homogeneity')[0, 0])
            Dissimilarity_1_pi_4.append(graycoprops(GLCM_1_pi_4, 'dissimilarity')[0, 0])
            Contrast_1_pi_4.append(graycoprops(GLCM_1_pi_4, 'contrast')[0, 0])

            GLCM_7_pi_2 = graycomatrix(self.Images[File], [7], [np.pi/2])
            Energy_7_pi_2.append(graycoprops(GLCM_7_pi_2, 'energy')[0, 0])
            Correlation_7_pi_2.append(graycoprops(GLCM_7_pi_2, 'correlation')[0, 0])
            Homogeneity_7_pi_2.append(graycoprops(GLCM_7_pi_2, 'homogeneity')[0, 0])
            Dissimilarity_7_pi_2.append(graycoprops(GLCM_7_pi_2, 'dissimilarity')[0, 0])
            Contrast_7_pi_2.append(graycoprops(GLCM_7_pi_2, 'contrast')[0, 0])

            GLCM_7_3_pi_4 = graycomatrix(self.Images[File], [7], [3 * np.pi/4])
            Energy_7_3_pi_4.append(graycoprops(GLCM_7_3_pi_4, 'energy')[0, 0])
            Correlation_7_3_pi_4.append(graycoprops(GLCM_7_3_pi_4, 'correlation')[0, 0])
            Homogeneity_7_3_pi_4.append(graycoprops(GLCM_7_3_pi_4, 'homogeneity')[0, 0])
            Dissimilarity_7_3_pi_4.append(graycoprops(GLCM_7_3_pi_4, 'dissimilarity')[0, 0])
            Contrast_7_3_pi_4.append(graycoprops(GLCM_7_3_pi_4, 'contrast')[0, 0])
          
            Labels.append(self.Label[0])
            # np.pi/4
            # np.pi/2
            # 3*np.pi/4

        except OSError:
            print('Cannot convert %s ❌' % File)
    
    # * Return the new dataframe with the new data
    DataFrame = pd.DataFrame({'Energy':Energy_1_0,  'Homogeneity':Homogeneity_1_0,  'Contrast':Contrast_1_0,  'Correlation':Correlation_1_0,
                            'Energy2':Energy_1_pi_4, 'Homogeneity2':Homogeneity_1_pi_4, 'Contrast2':Contrast_1_pi_4, 'Correlation2':Correlation_1_pi_4, 
                            'Energy3':Energy_7_pi_2, 'Homogeneity3':Homogeneity_7_pi_2, 'Contrast3':Contrast_7_pi_2, 'Correlation3':Correlation_7_pi_2, 
                            'Energy4':Energy_7_3_pi_4, 'Homogeneity4':Homogeneity_7_3_pi_4, 'Contrast4':Contrast_7_3_pi_4, 'Correlation4':Correlation_7_3_pi_4, 'Labels':Labels})


    #'Energy':Energy
    #'Homogeneity':Homogeneity
    #'Correlation':Correlation
    #'Contrast':Contrast
    #'Dissimilarity':Dissimilarity

    # * Return a dataframe with only the data without the labels
    X = DataFrame.iloc[:, [0, 1]].values

    # * Return a dataframe with only the labels
    Y = DataFrame.iloc[:, -1].values

    # * Return the three dataframes
    return DataFrame, X, Y, Glcm