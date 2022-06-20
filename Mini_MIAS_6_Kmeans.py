import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glrlm import GLRLM
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans

from Mini_MIAS_2_General_Functions import SortImages

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
    
    self.folder = kwargs.get('folder', None)
    #self.newfolder = kwargs.get('newfolder', None)
    self.extension = kwargs.get('extension', None)
    self.label = kwargs.get('label', None)

    #if self.folder == None:
      #raise ValueError("Folder does not exist")

  # FOF features folder

  def TexturesFeatureFirstOrder(self):

    Fof = 'First Order Features'

    Mean = []
    Var = []
    Skew = []
    Kurtosis = []
    Energy = []
    Entropy = []
    Labels = []

    #Images = [] # Png Images
    Filename = [] 

    os.chdir(self.folder)

    sorted_files, images = SortImages(self.folder)
    count = 1

    for File in sorted_files:
      if File.endswith(self.extension): # Read png files

        try:
          filename, extension  = os.path.splitext(File)
          print(f"Working with {count} of {images} {extension} images, {filename} ------- {self.extension} ✅")
          count += 1

          Path_File = os.path.join(self.folder, File)
          Imagen = cv2.imread(Path_File)
          
          #mean = np.mean(Imagen)
          #std = np.std(Imagen)
          #entropy = shannon_entropy(Imagen)
          #kurtosis_ = kurtosis(Imagen, axis = None)
          #skew_ = skew(Imagen, axis = None)
          #labels = ["FOS_Mean","FOS_Variance","FOS_Median","FOS_Mode","FOS_Skewness",
          #"FOS_Kurtosis","FOS_Energy","FOS_Entropy","FOS_MinimalGrayLevel",
          #"FOS_MaximalGrayLevel","FOS_CoefficientOfVariation",
          #"FOS_10Percentile","FOS_25Percentile","FOS_75Percentile",
          #"FOS_90Percentile","FOS_HistogramWidth"]

          Features, Labels_ = fos(Imagen, None)

          Mean.append(Features[0])
          Var.append(Features[1])
          Skew.append(Features[4])
          Kurtosis.append(Features[5])
          Energy.append(Features[6])
          Entropy.append(Features[7])
          Labels.append(self.label)

          Filename.append(filename)
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

    Dataset = pd.DataFrame({'REFNUM':Filename, 'Mean':Mean, 'Var':Var, 'Kurtosis':Kurtosis, 'Energy':Energy, 'Skew':Skew, 'Entropy':Entropy, 'Labels':Labels})

    X = Dataset.iloc[:, [1, 2, 3, 4, 5, 6]].values
    Y = Dataset.iloc[:, 0].values

    return Dataset, X, Y

  # GLRLM features folder

  def TexturesFeatureGLRLM(self):

    Glrlm = 'Gray-Level Run Length Matrix'

    SRE = []  # Short Run Emphasis
    LRE  = [] # Long Run Emphasis
    GLU = []  # Grey Level Uniformity
    RLU = []  # Run Length Uniformity
    RPC = []  # Run Percentage
    Labels = []
    #Images = [] # Png Images
    Filename = [] 

    os.chdir(self.folder)

    sorted_files, images = SortImages(self.folder)
    count = 1

    for File in sorted_files:
      if File.endswith(self.extension): # Read png files

        try:
          filename, extension  = os.path.splitext(File)
          print(f"Working with {count} of {images} {extension} images, {filename} ------- {self.extension} ✅")
          count += 1

          Path_File = os.path.join(self.folder, File)
          Imagen = cv2.imread(Path_File)
          Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)

          app = GLRLM()
          glrlm = app.get_features(Imagen, 8)
          print(glrlm.Features)

          SRE.append(glrlm.Features[0])
          LRE.append(glrlm.Features[1])
          GLU.append(glrlm.Features[2])
          RLU.append(glrlm.Features[3])
          RPC.append(glrlm.Features[4])
          Labels.append(self.label)

          Filename.append(filename)
          #Extensions.append(extension)

        except OSError:
          print('Cannot convert %s ❌' % File)

    Dataset = pd.DataFrame({'REFNUM':Filename, 'SRE':SRE, 'LRE':LRE, 'GLU':GLU, 'RLU':RLU, 'RPC':RPC, 'Labels':Labels})

    X = Dataset.iloc[:, [1, 2, 3, 4, 5]].values
    Y = Dataset.iloc[:, -1].values

    return Dataset, X, Y

  # GLCM features folder

  def TexturesFeatureGLCM(self):
    
    Glcm = 'Gray-Level Co-Occurance Matrix'

    Dissimilarity = []
    Correlation = []
    Homogeneity = []
    Energy = []
    Contrast = []
    ASM = []
    Labels = []
    #Images = [] # Png Images
    Filename = [] 

    os.chdir(self.folder)

    sorted_files, images = SortImages(self.folder)
    count = 1

    for File in sorted_files:
      if File.endswith(self.extension): # Read png files

        try:
          filename, extension  = os.path.splitext(File)
          print(f"Working with {count} of {images} {extension} images, {filename} ------- {self.extension} ✅")
          count += 1

          Path_File = os.path.join(self.folder, File)
          Imagen = cv2.imread(Path_File)
          Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)

          GLCM = graycomatrix(Imagen, [1], [0, np.pi/4, np.pi/2, 3 * np.pi/4])
          Energy.append(graycoprops(GLCM, 'energy')[0, 0])
          Correlation.append(graycoprops(GLCM, 'correlation')[0, 0])
          Homogeneity.append(graycoprops(GLCM, 'homogeneity')[0, 0])
          Dissimilarity.append(graycoprops(GLCM, 'dissimilarity')[0, 0])
          Contrast.append(graycoprops(GLCM, 'contrast')[0, 0])
          ASM.append(graycoprops(GLCM, 'ASM')[0, 0])
          Labels.append(self.label)

          Filename.append(filename)
          #Extensions.append(extension)

        except OSError:
          print('Cannot convert %s ❌' % File)

    Dataset = pd.DataFrame({'REFNUM':Filename, 'Energy':Energy, 'Correlation':Correlation, 'Homogeneity':Homogeneity, 'Dissimilarity':Dissimilarity, 'Contrast':Contrast, 'ASM':ASM, 'Labels':Labels})

    X = Dataset.iloc[:, [1, 2, 3, 4, 5, 6]].values
    Y = Dataset.iloc[:, 0].values

    return Dataset, X, Y

  # FOF features images

  def TexturesFeatureFirstOrderImage(Images, Label):

    Fof = 'First Order Features'

    Mean = []
    Var = []
    Skew = []
    Kurtosis = []
    Energy = []
    Entropy = []
    Labels = []

    count = 1

    for File in range(len(Images)):

        try:

            print(f"Working with {count} of {len(Images)} images ✅")
            count += 1

            Images[File] = cv2.cvtColor(Images[File], cv2.COLOR_BGR2GRAY)

            Features, Labels_ = fos(Images[File], None)

            Mean.append(Features[0])
            Var.append(Features[1])
            Skew.append(Features[4])
            Kurtosis.append(Features[5])
            Energy.append(Features[6])
            Entropy.append(Features[7])
            Labels.append(Label)

        except OSError:
            print('Cannot convert %s ❌' % File)

    Dataset = pd.DataFrame({'Mean':Mean, 'Var':Var, 'Kurtosis':Kurtosis, 'Energy':Energy, 'Skew':Skew, 'Entropy':Entropy, 'Labels':Labels})

    X = Dataset.iloc[:, [0, 1, 2, 3, 4, 5]].values
    Y = Dataset.iloc[:, -1].values

    return Dataset, X, Y, Fof

  # GLRLM features images

  def TexturesFeatureGLRLMImage(Images, Label):

    Glrlm = 'Gray-Level Run Length Matrix'

    SRE = []  # Short Run Emphasis
    LRE  = [] # Long Run Emphasis
    GLU = []  # Grey Level Uniformity
    RLU = []  # Run Length Uniformity
    RPC = []  # Run Percentage
    Labels = []

    count = 1

    for File in range(len(Images)):

        try:
            print(f"Working with {count} of {len(Images)} images ✅")
            count += 1

            app = GLRLM()
            glrlm = app.get_features(Images[File], 8)

            SRE.append(glrlm.Features[0])
            LRE.append(glrlm.Features[1])
            GLU.append(glrlm.Features[2])
            RLU.append(glrlm.Features[3])
            RPC.append(glrlm.Features[4])
            Labels.append(Label)

        except OSError:
            print('Cannot convert %s ❌' % File)

    Dataset = pd.DataFrame({'SRE':SRE, 'LRE':LRE, 'GLU':GLU, 'RLU':RLU, 'RPC':RPC, 'Labels':Labels})

    X = Dataset.iloc[:, [0, 1, 2, 3, 4]].values
    Y = Dataset.iloc[:, -1].values

    return Dataset, X, Y, Glrlm

  # GLCM features images

  def TexturesFeatureGLCMImage(Images, Label):

    Glcm = 'Gray-Level Co-Occurance Matrix'

    Dataset = pd.DataFrame()
    
    Dissimilarity = []
    Correlation = []
    Homogeneity = []
    Energy = []
    Contrast = []

    Dissimilarity2 = []
    Correlation2 = []
    Homogeneity2 = []
    Energy2 = []
    Contrast2 = []

    Dissimilarity3 = []
    Correlation3 = []
    Homogeneity3 = []
    Energy3 = []
    Contrast3 = []

    Dissimilarity4 = []
    Correlation4 = []
    Homogeneity4 = []
    Energy4 = []
    Contrast4 = []

    #Entropy = []
    #ASM = []
    Labels = []
    Labels2 = []
    Labels3 = []
    Labels4 = []

    count = 1

    for File in range(len(Images)):

        try:
            print(f"Working with {count} of {len(Images)} images ✅")
            count += 1

            Images[File] = cv2.cvtColor(Images[File], cv2.COLOR_BGR2GRAY)

            GLCM = graycomatrix(Images[File], [1], [0])
            Energy.append(graycoprops(GLCM, 'energy')[0, 0])
            Correlation.append(graycoprops(GLCM, 'correlation')[0, 0])
            Homogeneity.append(graycoprops(GLCM, 'homogeneity')[0, 0])
            Dissimilarity.append(graycoprops(GLCM, 'dissimilarity')[0, 0])
            Contrast.append(graycoprops(GLCM, 'contrast')[0, 0])
          
            GLCM2 = graycomatrix(Images[File], [1], [np.pi/4])
            Energy2.append(graycoprops(GLCM2, 'energy')[0, 0])
            Correlation2.append(graycoprops(GLCM2, 'correlation')[0, 0])
            Homogeneity2.append(graycoprops(GLCM2, 'homogeneity')[0, 0])
            Dissimilarity2.append(graycoprops(GLCM2, 'dissimilarity')[0, 0])
            Contrast2.append(graycoprops(GLCM2, 'contrast')[0, 0])

            GLCM3 = graycomatrix(Images[File], [7], [np.pi/2])
            Energy3.append(graycoprops(GLCM3, 'energy')[0, 0])
            Correlation3.append(graycoprops(GLCM3, 'correlation')[0, 0])
            Homogeneity3.append(graycoprops(GLCM3, 'homogeneity')[0, 0])
            Dissimilarity3.append(graycoprops(GLCM3, 'dissimilarity')[0, 0])
            Contrast3.append(graycoprops(GLCM3, 'contrast')[0, 0])

            GLCM4 = graycomatrix(Images[File], [7], [3 * np.pi/4])
            Energy4.append(graycoprops(GLCM4, 'energy')[0, 0])
            Correlation4.append(graycoprops(GLCM4, 'correlation')[0, 0])
            Homogeneity4.append(graycoprops(GLCM4, 'homogeneity')[0, 0])
            Dissimilarity4.append(graycoprops(GLCM4, 'dissimilarity')[0, 0])
            Contrast4.append(graycoprops(GLCM4, 'contrast')[0, 0])
          
            Labels.append(Label)
            # np.pi/4
            # np.pi/2
            # 3*np.pi/4

        except OSError:
            print('Cannot convert %s ❌' % File)
    

    Dataset = pd.DataFrame({'Energy':Energy,  'Homogeneity':Homogeneity,  'Contrast':Contrast,  'Correlation':Correlation,
                            'Energy2':Energy2, 'Homogeneity2':Homogeneity2, 'Contrast2':Contrast2, 'Correlation2':Correlation2, 
                            'Energy3':Energy3, 'Homogeneity3':Homogeneity3, 'Contrast3':Contrast3, 'Correlation3':Correlation3, 
                            'Energy4':Energy4, 'Homogeneity4':Homogeneity4, 'Contrast4':Contrast4, 'Correlation4':Correlation4, 'Labels':Labels})


    #'Energy':Energy
    #'Homogeneity':Homogeneity
    #'Correlation':Correlation
    #'Contrast':Contrast
    #'Dissimilarity':Dissimilarity

    X = Dataset.iloc[:, [0, 1]].values
    Y = Dataset.iloc[:, -1].values

    return Dataset, X, Y, Glcm

# class about Kmeans algorithm

class Kmeans:

  def __init__(self, **kwargs):

    self.folder = kwargs.get('folder', None)
    self.folderpic = kwargs.get('folderpic', None)
    self.folderCSV = kwargs.get('folderCSV', None)
    self.severity = kwargs.get('severity', None)
    self.name = kwargs.get('name', None)
    self.X = kwargs.get('X', None)
    self.clusters = kwargs.get('clusters', None)
    self.filename = kwargs.get('filename', None)
    self.df = kwargs.get('df', None)
    self.removecluster = kwargs.get('CR', None)

  def KmeansFunction(self):

    """
	  Using the elbow method and get k-means clusters.

    Parameters:
    argument1 (List): Data that will be cluster
    argument2 (int): How many cluster will be use

    Returns:
	  model:Returning kmeans model
    list:Returning kmeans y axis
    int:Returning number of clusters
   	"""
    Colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'indigo', 'azure', 'tan', 'purple']

    wcss = []
    for i in range(1, 10):
      kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
      kmeans.fit(self.X)
      wcss.append(kmeans.inertia_)
    plt.plot(range(1, 10), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    #plt.show()

    kmeans = KMeans(n_clusters = self.clusters, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(self.X)

    for i in range(self.clusters):

      if  self.clusters <= 10:

          plt.scatter(self.X[y_kmeans == i, 0], self.X[y_kmeans == i, 1], s = 100, c = Colors[i], label = 'Cluster ' + str(i))


    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'yellow', label = 'Centroids')

    plt.title('Clusters')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend()

    dst = 'Kmeans_Graph_' + str(self.name) + '_' + str(self.severity) + '.png'
    dstPath = os.path.join(self.folderpic, dst)

    plt.savefig(dstPath)

    #plt.show()

    DataFrame = pd.DataFrame({'y_kmeans':y_kmeans, 'REFNUM':self.filename})
    pd.set_option('display.max_rows', DataFrame.shape[0] + 1)
    #print(DataFrame)

    pd.set_option('display.max_rows', DataFrame.shape[0] + 1)
    dst = str(self.name) + '_Dataframe_' + str(self.severity) + '.csv'
    dstPath = os.path.join(self.folderCSV, dst)

    DataFrame.to_csv(dstPath)

    #print(DataFrame['y_kmeans'].value_counts())

    return DataFrame

  # Remove Data from K-means function

  def RemoveDataKmeans(self):

    """
	  Remove the cluster chosen from dataframe

    Parameters:
    argument1 (Folder): Folder's path
    argument2 (dataframe): dataframe that will be used to remove data
    argument3 (int): the cluster's number that will be remove

    Returns:
	  dataframe:Returning dataframe already modified
    """

    #Images = [] # Png Images
    Filename = [] 
    DataRemove = []
    Data = 0

    KmeansValue = 0
    Refnum = 1

    os.chdir(self.folder)

    sorted_files, images = SortImages(self.folder)
    count = 1
    Index = 1

    for File in sorted_files:

      filename, extension  = os.path.splitext(File)

      if self.df.iloc[Index - 1, Refnum] == filename: # Read png files

        print(filename)
        print(self.df.iloc[Index - 1, Refnum])

        if self.df.iloc[Index - 1, KmeansValue] == self.removecluster:

          try:
            print(f"Working with {count} of {images} {extension} images, {filename} ------- {extension} ✅")
            count += 1

            Path_File = os.path.join(self.folder, File)
            #print(Path_File)
            os.remove(Path_File)
            print(self.df.iloc[Index - 1, Refnum], ' removed ❌')
            DataRemove.append(count)
            Data += 0
            #df = df.drop(df.index[count])

          except OSError:
            print('Cannot convert %s ❌' % File)

        elif self.df.iloc[Index - 1, KmeansValue] != self.removecluster:
        
          Filename.append(filename)

        Index += 1

      elif self.df.iloc[Index - 1, Refnum] != filename:
      
        print(filename)
        print(self.df.iloc[Index - 1, Refnum])
        print('Files are not equal')
        break

      else:
  
        Index += 1

      for i in range(Data):

        self.df = self.df.drop(self.df.index[DataRemove[i]])

  #Dataset = pd.DataFrame({'y_kmeans':df_u.iloc[Index - 1, REFNUM], 'REFNUM':df_u.iloc[Index - 1, KmeansValue]})
  #X = Dataset.iloc[:, [0, 1, 2, 3, 4]].values

    #print(df)

    pd.set_option('display.max_rows', self.df.shape[0] + 1)

    dst = str(self.name) + '_Data_Removed_' + str(self.severity) + '.csv'
    dstPath = os.path.join(self.folderCSV, dst)

    self.df.to_csv(dstPath)

    return self.df