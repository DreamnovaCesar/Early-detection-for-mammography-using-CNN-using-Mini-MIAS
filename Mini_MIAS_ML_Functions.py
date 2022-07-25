import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import cycle

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.multiclass import OneVsRestClassifier

#from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE

def Machine_learning_config(Dataframe, Dataframe_save, Folder_path, Column_names, ML_model, Enhancement_technique, Extract_feature_technique, Class_labels, Folder_data, Folder_models):

    """
	  Extract features from each image, it could be FOS, GLCM or GRLM.

    Parameters:
    argument1 (dataframe): Datraframe without values.
    argument2 (list): Values of each model
    argument3 (folder): Folder to save images with metadata
    argument4 (folder): Folder to save images with metadata (CSV)
    argument5 (str): Name of each model

    Returns:
	  list:Returning all metadata of each model trained.
    
   	"""
    sm = SMOTE()
    sc = StandardScaler()
    #ALL_ML_model = len(ML_model)

    for Index, Model in enumerate(ML_model):

        # * Class problem definition
        Class_problem = len(Class_labels)

        if Class_problem == 2:
            Class_problem_prefix = 'Biclass_'
        elif Class_problem >= 3:
            Class_problem_prefix = 'Multiclass_'

        # * Extract data and label
        Dataframe_len_columns = len(Dataframe.columns)

        X_total = Dataframe.iloc[:, 0:Dataframe_len_columns - 1].values
        Y_total = Dataframe.iloc[:, -1].values

        print(X_total)
        print(Y_total)

        pd.set_option('display.max_rows', Dataframe.shape[0] + 1)
        print(Dataframe)

        # * Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(np.array(X_total), np.array(Y_total), test_size = 0.2, random_state = 1)

        # * Resample data for training
        X_train, y_train = sm.fit_resample(X_train, y_train)

        # * Scaling data for training
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        print(y_train)
        print(y_test)

        # * Chose the machine learning.
        Info_model = Machine_learning_models(Model, Enhancement_technique, Class_labels, X_train, y_train, X_test, y_test, Folder_models)
    
        Overwrite_row_CSV(Dataframe_save, Folder_path, Info_model, Column_names, Index)

    # * Save dataframe in the folder given
    Class_problem_dataframe = str(Class_problem_prefix) + 'Dataframe_' + str(Extract_feature_technique) + '_' + str(Enhancement_technique) + '.csv'
    Class_problem_folder = os.path.join(Folder_data, Class_problem_dataframe)
    Dataframe.to_csv(Class_problem_folder)

    return Info_model

# ?

def Machine_learning_models(ML_model, Enhancement_technique, Class_labels, X_train, y_train, X_test, y_test, Folder_models):

    """
	  General configuration for each model, extracting features and printing theirs values (Machine Learning).

    Parameters:
    argument1 (model): Model chosen.
    argument2 (str): technique used.
    argument3 (list): labels used for printing.
    argument4 (int): X train split data.
    argument5 (int): y train split data.
    argument6 (int): X test split data.
    argument7 (int): y test split data.
    argument8 (int): Folder used to save data images.
    argument9 (int): Folder used to save data images in spanish.

    Returns:
	  int:Returning all data from each model.
    
   	"""
    # * Parameters plt

    Height = 5
    Width = 12
    Annot_kws = 12
    Font = 0.7
    H = 0.02
    
    X_size_figure = 1
    Y_size_figure = 2

    # * Parameters dic classification report

    Macro_avg_label = 'macro avg'
    Weighted_avg_label = 'weighted avg'

    Classification_report_labels = []
    Classification_report_metrics_labels = ('precision', 'recall', 'f1-score', 'support')

    for Label in Class_labels:
      Classification_report_labels.append(Label)
    
    Classification_report_labels.append(Macro_avg_label)
    Classification_report_labels.append(Weighted_avg_label)

    Classification_report_values = []

    # * Metrics digits

    Digits = 4

    # * Lists

    Info = []
    Labels_multiclass = []
    
    # * Class problem definition
    Class_problem = len(Class_labels)

    # * Conditional if the class problem was biclass or multiclass

    if Class_problem == 2:
        Class_problem_prefix = 'Biclass_'
    elif Class_problem > 2:
        Class_problem_prefix = 'Multiclass_'

    if len(Class_labels) == 2:

        # * Get the data from the model chosen
        Y_pred, Total_time_training, Model_name = ML_model(X_train, y_train, X_test)

        # * Confusion Matrix
        Confusion_matrix = confusion_matrix(y_test, Y_pred)
        #cf_matrix = confusion_matrix(y_test, y_pred)

        print(Confusion_matrix)
        print(classification_report(y_test, Y_pred, target_names = Class_labels))

        Dict = classification_report(y_test, Y_pred, target_names = Class_labels, output_dict = True)

        for i, Report_labels in enumerate(Classification_report_labels):
            for i, Metric_labels in enumerate(Classification_report_metrics_labels):

                print(Dict[Report_labels][Metric_labels])
                Classification_report_values.append(Dict[Report_labels][Metric_labels])
                print("\n")

        # * Accuracy
        Accuracy = accuracy_score(y_test, Y_pred)
        print(f"Accuracy: {round(Accuracy, Digits)}")
        print("\n")

        # * Precision
        Precision = precision_score(y_test, Y_pred)
        print(f"Precision: {round(Precision, Digits)}")
        print("\n")

        # * Recall
        Recall = recall_score(y_test, Y_pred)
        print(f"Recall: {round(Recall, Digits)}")
        print("\n")

        # * F1-score
        F1_Score = f1_score(y_test, Y_pred)
        print(f"F1: {round(F1_Score, Digits)}")
        print("\n")

        Confusion_matrix_dataframe = pd.DataFrame(Confusion_matrix, range(len(Confusion_matrix)), range(len(Confusion_matrix[0])))

        # * Figure's size
        plt.figure(figsize = (Width, Height))
        plt.subplot(X_size_figure, Y_size_figure, 1)
        sns.set(font_scale = Font) # for label size

        # * Confusion matrix heatmap
        ax = sns.heatmap(Confusion_matrix_dataframe, annot = True, fmt = 'd')
        #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values')
        ax.set_xticklabels(Class_labels)
        ax.set_yticklabels(Class_labels)

        # * FPR and TPR values for the ROC curve
        FPR, TPR, _ = roc_curve(y_test, Y_pred)
        Auc = auc(FPR, TPR)
        
        # * Subplot ROC curve
        plt.subplot(X_size_figure, Y_size_figure, 2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(FPR, TPR, label = Model_name + '(area = {:.4f})'.format(Auc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc = 'lower right')

        # * Save this figure in the folder given
        Class_problem_name = Class_problem_prefix + str(Model_name) + str(Enhancement_technique) + '_' + '.png'
        Class_problem_folder = os.path.join(Folder_models, Class_problem_name)

        plt.savefig(Class_problem_folder)
        #plt.show()

    elif len(Class_labels) >= 3:

        # * Get the data from the model chosen
        Y_pred, Total_time_training, Model_name = ML_model(X_train, y_train, X_test)

        # * Binarize labels to get multiples ROC curves
        for i in range(len(Class_labels)):
            Labels_multiclass.append(i)
        
        print(Y_pred)

        y_pred_roc = label_binarize(Y_pred, classes = Labels_multiclass)
        y_test_roc = label_binarize(y_test, classes = Labels_multiclass)

        # * Confusion Matrix
        Confusion_matrix = confusion_matrix(y_test, Y_pred)
        #cf_matrix = confusion_matrix(y_test, y_pred)

        print(confusion_matrix(y_test, Y_pred))
        print(classification_report(y_test, Y_pred, target_names = Class_labels))

        # * Accuracy
        Accuracy = accuracy_score(y_test, Y_pred)
        print(f"Precision: {round(Accuracy, Digits)}")
        print("\n")

        # * Precision
        Precision = precision_score(y_test, Y_pred, average = 'weighted')
        print(f"Precision: {round(Precision, Digits)}")
        print("\n")

        # * Recall
        Recall = recall_score(y_test, Y_pred, average = 'weighted')
        print(f"Recall: {round(Recall, Digits)}")
        print("\n")

        # * F1-score
        F1_Score = f1_score(y_test, Y_pred, average = 'weighted')
        print(f"F1: {round(F1_Score, Digits)}")
        print("\n")

        Confusion_matrix_dataframe = pd.DataFrame(Confusion_matrix, range(len(Confusion_matrix)), range(len(Confusion_matrix[0])))

        # * Figure's size
        plt.figure(figsize = (Width, Height))
        plt.subplot(X_size_figure, Y_size_figure, 1)
        sns.set(font_scale = Font) # for label size

        # * Confusion matrix heatmap
        ax = sns.heatmap(Confusion_matrix_dataframe, annot = True, fmt = 'd') # font size
        #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values')
        ax.set_xticklabels(Class_labels)
        ax.set_yticklabels(Class_labels)

        # * FPR and TPR values for the ROC curve
        FPR = dict()
        TPR = dict()
        Roc_auc = dict()

        for i in range(Class_problem):
            FPR[i], TPR[i], _ = roc_curve(y_test_roc[:, i], y_pred_roc[:, i])
            Roc_auc[i] = auc(FPR[i], TPR[i])

        colors = cycle(['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan'])
        
        # * Subplot several ROC curves
        plt.subplot(X_size_figure, Y_size_figure, 2)
        plt.plot([0, 1], [0, 1], 'k--')

        for i, color, lbl in zip(range(Class_problem), colors, Class_labels):
            plt.plot(FPR[i], TPR[i], color = color, label = 'ROC Curve of class {0} (area = {1:0.4f})'.format(lbl, Roc_auc[i]))

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc = 'lower right')

        # * Save this figure in the folder given
        Class_problem_name = Class_problem_prefix + str(Model_name) + str(Enhancement_technique) + '.png'
        Class_problem_folder = os.path.join(Folder_models, Class_problem_name)

        plt.savefig(Class_problem_folder)
        #plt.show()
    
    Info.append(Model_name + '_' + Enhancement_technique)
    Info.append(Model_name)
    Info.append(Accuracy)
    Info.append(Precision)
    Info.append(Recall)
    Info.append(F1_Score)
    Info.append(len(y_train))
    Info.append(len(y_test))
    Info.append(Total_time_training)
    Info.append(Enhancement_technique)
    Info.append(Confusion_matrix[0][0])
    Info.append(Confusion_matrix[0][1])
    Info.append(Confusion_matrix[1][0])
    Info.append(Confusion_matrix[1][1])

    if Class_problem == 2:
        Info.append(Auc)
    elif Class_problem >= 3:
        for i in range(Class_problem):
            Info.append(Roc_auc[i])

    return Info

# ?

def Overwrite_row_CSV(Dataframe, Folder_path, Info_list, Column_names, Row):

    """
	  Updates final CSV dataframe to see all values

    Parameters:
    argument1 (list): All values.
    argument2 (dataframe): dataframe that will be updated
    argument3 (list): Names of each column
    argument4 (folder): Folder path to save the dataframe
    argument5 (int): The index.

    Returns:
	  void
    
   	"""

    for i in range(len(Info_list)):
        Dataframe.loc[Row, Column_names[i]] = Info_list[i]
  
    Dataframe.to_csv(Folder_path, index = False)
  
    print(Dataframe)

# ?

def SVM(X_train, y_train, X_test):

    """
	  SVM configuration.

    Parameters:
    argument1 (int): X train split data.
    argument2 (int): y train split data.
    argument3 (int): X test split data.

    Returns:
	    list:Model's prediction list
        int:Training's time
        int:Model's name
        funcition:Classifier


   	"""

    Model_name = 'SVM'

    Start_training_time = time.time()
    
    # Data Custom model
    classifier = SVC(kernel = 'rbf', C = 1)
    classifier.fit(X_train, y_train)

    End_training_time = time.time()

    Total_time_training = End_training_time - Start_training_time
    
    Y_pred = classifier.predict(X_test)

    return Y_pred, Total_time_training, Model_name

# ?

def Multi_SVM(X_train, y_train, X_test):

    """
	  MultiSVM configuration.

    Parameters:
    argument1 (int): X train split data.
    argument2 (int): y train split data.
    argument3 (int): X test split data.

    Returns:
	    list:Model's prediction list
        int:Training's time
        int:Model's name
        funcition:Classifier
    
   	"""

    Model_name = 'Multi SVM'

    Start_training_time = time.time()
    
    # Data Custom model
    classifier = OneVsRestClassifier(SVC(kernel = 'rbf', C = 1))
    classifier.fit(X_train, y_train)

    End_training_time = time.time()

    Total_time_training = End_training_time - Start_training_time
    
    Y_pred = classifier.predict(X_test)

    return Y_pred, Total_time_training, Model_name

# ?

def MLP(X_train, y_train, X_test):

    """
	  MLP configuration.

    Parameters:
    argument1 (int): X train split data.
    argument2 (int): y train split data.
    argument3 (int): X test split data.

    Returns:
	    list:Model's prediction list
        int:Training's time
        int:Model's name
        funcition:Classifier


   	"""

    Model_name = 'MLP'

    Start_training_time = time.time()
    
    # Data Custom model
    classifier = MLPClassifier(hidden_layer_sizes = [100] * 2, random_state = 1, max_iter = 2000)
    classifier.fit(X_train, y_train)

    End_training_time = time.time()

    Total_time_training = End_training_time - Start_training_time
    
    Y_pred = classifier.predict(X_test)

    return Y_pred, Total_time_training, Model_name

# ?

def DT(X_train, y_train, X_test):

    """
	  Decision Tree configuration.

    Parameters:
    argument1 (int): X train split data.
    argument2 (int): y train split data.
    argument3 (int): X test split data.

    Returns:
	    list:Model's prediction list
        int:Training's time
        int:Model's name
        funcition:Classifier
    
   	"""

    Model_name = 'DT'

    Start_training_time = time.time()
    
    # Data Custom model
    classifier = DecisionTreeClassifier(max_depth = 50)
    classifier.fit(X_train, y_train)

    End_training_time = time.time()

    Total_time_training = End_training_time - Start_training_time
    
    Y_pred = classifier.predict(X_test)

    return Y_pred, Total_time_training, Model_name

# ?

def KNN(X_train, y_train, X_test):

    """
	  KNeighbors configuration.

    Parameters:
    argument1 (int): X train split data.
    argument2 (int): y train split data.
    argument3 (int): X test split data.

    Returns:
	    list:Model's prediction list
        int:Training's time
        int:Model's name
        funcition:Classifier
    
   	"""

    Model_name = 'KNN'

    Start_training_time = time.time()
    
    # Data Custom model
    classifier = KNeighborsClassifier(n_neighbors = 7)
    classifier.fit(X_train, y_train)

    End_training_time = time.time()

    Total_time_training = End_training_time - Start_training_time
    
    Y_pred = classifier.predict(X_test)

    return Y_pred, Total_time_training, Model_name

# ?

def RF(X_train, y_train, X_test):

    """
	  Random forest configuration.

    Parameters:
    argument1 (int): X train split data.
    argument2 (int): y train split data.
    argument3 (int): X test split data.

    Returns:
	    list:Model's prediction list
        int:Training's time
        int:Model's name
        funcition:Classifier

   	"""

    Model_name = 'RF'

    Start_training_time = time.time()
    
    # Data Custom model
    classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)

    End_training_time = time.time()

    Total_time_training = End_training_time - Start_training_time
    
    Y_pred = classifier.predict(X_test)

    return Y_pred, Total_time_training, Model_name

# ?

def GBC(X_train, y_train, X_test):

    """
	  Random forest configuration.

    Parameters:
    argument1 (int): X train split data.
    argument2 (int): y train split data.
    argument3 (int): X test split data.

    Returns:
	    list:Model's prediction list
        int:Training's time
        int:Model's name
        funcition:Classifier

   	"""
       
    Model_name = 'GBC'

    Start_training_time = time.time()
    
    # Data Custom model
    classifier = GradientBoostingClassifier(n_estimators = 100, learning_rate = 1.0, max_depth = 2, random_state = 0)
    classifier.fit(X_train, y_train)

    End_training_time = time.time()

    Total_time_training = End_training_time - Start_training_time
    
    Y_pred = classifier.predict(X_test)

    return Y_pred, Total_time_training, Model_name
