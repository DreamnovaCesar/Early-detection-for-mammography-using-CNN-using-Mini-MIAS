import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from Mini_MIAS_2_General_Functions import sort_images

  # ? Kmeans algorithm

def kmeans_function(CSV_folder, GRAPH_folder, Technique_name, X_data, Clusters, Filename, Severity):

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

    # * Tuple with different colors
    Colors = ('red', 'blue', 'green', 'cyan', 'magenta', 'indigo', 'azure', 'tan', 'purple')

    wcss = []
    for i in range(1, 10):
      kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
      kmeans.fit(X_data)
      wcss.append(kmeans.inertia_)
    plt.plot(range(1, 10), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    #plt.show()

    kmeans = KMeans(n_clusters = Clusters, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(X_data)

    for i in range(Clusters):

      if  Clusters <= 10:

          plt.scatter(X_data[y_kmeans == i, 0], X_data[y_kmeans == i, 1], s = 100, c = Colors[i], label = 'Cluster ' + str(i))


    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'yellow', label = 'Centroids')

    plt.title('Clusters')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend()

    GRAPH_name = 'Kmeans_Graph_' + str(Technique_name) + '_' + str(Severity) + '.png'
    GRAPH_folder = os.path.join(GRAPH_folder, GRAPH_name)

    plt.savefig(GRAPH_folder)

    #plt.show()

    DataFrame = pd.DataFrame({'y_kmeans' : y_kmeans, 'REFNUM' : Filename})
    #pd.set_option('display.max_rows', DataFrame.shape[0] + 1)
    #print(DataFrame)

    #pd.set_option('display.max_rows', DataFrame.shape[0] + 1)
    Dataframe_name = str(Technique_name) + '_Dataframe_' + str(Severity) + '.csv'
    Dataframe_folder = os.path.join(CSV_folder, Dataframe_name)

    DataFrame.to_csv(Dataframe_folder)

    #print(DataFrame['y_kmeans'].value_counts())

    return DataFrame

  # ? Remove Data from K-means function

def kmeans_remove_data(Folder, CSV_folder, Technique_name, Dataframe, Cluster_to_remove, Severity):

    """
	  Remove the cluster chosen from dataframe

    Parameters:
    argument1 (Folder): Folder's path
    argument2 (dataframe): dataframe that will be used to remove data
    argument3 (int): the cluster's number that will be remove

    Returns:
	  dataframe:Returning dataframe already modified
    """

    # * General lists
    #Images = [] # Png Images
    All_filename = [] 

    DataRemove = []
    Data = 0

    KmeansValue = 0
    Refnum = 1
    count = 1
    Index = 1

    os.chdir(Folder)

    # * Using sort function
    sorted_files, images = sort_images(Folder)

    # * Reading the files
    for File in sorted_files:

      Filename, Format = os.path.splitext(File)

      if Dataframe.iloc[Index - 1, Refnum] == Filename: # Read png files

        print(Filename)
        print(Dataframe.iloc[Index - 1, Refnum])

        if Dataframe.iloc[Index - 1, KmeansValue] == Cluster_to_remove:

          try:
            print(f"Working with {count} of {images} {Format} images, {Filename} ------- {Format} ✅")
            count += 1

            Path_File = os.path.join(Folder, File)
            os.remove(Path_File)
            print(Dataframe.iloc[Index - 1, Refnum], ' removed ❌')
            DataRemove.append(count)
            Data += 0

            #df = df.drop(df.index[count])

          except OSError:
            print('Cannot convert %s ❌' % File)

        elif Dataframe.iloc[Index - 1, KmeansValue] != Cluster_to_remove:
        
          All_filename.append(Filename)

        Index += 1

      elif Dataframe.iloc[Index - 1, Refnum] != Filename:
      
        print(Dataframe.iloc[Index - 1, Refnum]  + '----' + Filename)
        print(Dataframe.iloc[Index - 1, Refnum])
        raise ValueError("Files are not the same") #! Alert

      else:
  
        Index += 1

      for i in range(Data):

        Dataframe = Dataframe.drop(Dataframe.index[DataRemove[i]])

  #Dataset = pd.DataFrame({'y_kmeans':df_u.iloc[Index - 1, REFNUM], 'REFNUM':df_u.iloc[Index - 1, KmeansValue]})
  #X = Dataset.iloc[:, [0, 1, 2, 3, 4]].values

    #print(df)
    #pd.set_option('display.max_rows', df.shape[0] + 1)

    Dataframe_name = str(Technique_name) + '_Data_Removed_' + str(Severity) + '.csv'
    Dataframe_folder = os.path.join(CSV_folder, Dataframe_name)

    Dataframe.to_csv(Dataframe_folder)

    return Dataframe