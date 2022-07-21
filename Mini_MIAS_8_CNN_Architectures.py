import os
import sys
import time
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from itertools import cycle

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.datasets import make_classification
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras import Input
from tensorflow.keras import datasets
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import Xception

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization

from sklearn.model_selection import train_test_split

"""
class pretrainedModels:

  def __init__(self, **kwargs):
    
    self.folder = kwargs.get('folder', None)
    self.newfolder = kwargs.get('newfolder', None)
    self.extension = kwargs.get('extension', None)
    self.newextension = kwargs.get('newextension', None)
"""

# ? Configuration of each DCNN model

def configuration_models(All_images, All_labels, Dataframe_save, Folder_path, DL_model, Enhancement_technique, Class_labels, Column_names, X_size, Y_size, Vali_split, Epochs, Folder_data, Folder_models, Folder_models_esp):

    #print(X_size)
    #print(Y_size)

    for Index, Model in enumerate(DL_model):

      #print(All_images)
      #print(All_labels)

      #All_images[0] = np.array(All_images[0])
      #All_images[1] = np.array(All_images[1])

      #All_labels[0] = np.array(All_labels[0])
      #All_labels[1] = np.array(All_labels[1])

      #print(len(All_images[0]))
      #print(len(All_images[1]))

      #print(len(All_labels[0]))
      #print(len(All_labels[1]))

      #All_images_CNN = All_images[0] + All_images[1]
      #All_labels_CNN = np.concatenate((All_labels[0], All_labels[1]), axis = None)

      print(len(All_images))
      print(len(All_labels))

      X_train, X_test, y_train, y_test = train_test_split(np.array(All_images), np.array(All_labels), test_size = 0.20, random_state = 42)

      Info_model = deep_learning_models(Model, Enhancement_technique, Class_labels, X_size, Y_size, Vali_split, Epochs, X_train, y_train, X_test, y_test, Folder_models, Folder_models_esp)
      
      Info_dataframe = overwrite_row_CSV(Dataframe_save, Folder_path, Info_model, Column_names, Index)

    return Info_dataframe

# ? Pretrained model configurations

def deep_learning_models(Pretrained_model_function, Enhancement_technique, Class_labels, X_size, Y_size, Vali_split, Epochs, X_train, y_train, X_test, y_test, Folder_models, Folder_models_Esp):

    """
	  General configuration for each model, extracting features and printing theirs values.

    Parameters:
    argument1 (model): Model chosen.
    argument2 (str): technique used.
    argument3 (list): labels used for printing.
    argument4 (int): Size of X.
    argument5 (int): Size of Y.
    argument6 (int): Number of classes.
    argument7 (float): Validation split value.
    argument8 (int): Number of epochs.
    argument9 (int): X train split data.
    argument9 (int): y train split data.
    argument9 (int): X test split data.
    argument9 (int): y test split data.
    argument9 (int): Folder used to save data images.
    argument9 (int): Folder used to save data images in spanish.

    Returns:
	  int:Returning all metadata from each model.
    
   	"""

    # * Parameters plt

    Height = 12
    Width = 12
    Annot_kws = 12
    font = 0.7

    X_size_figure = 2
    Y_size_figure = 2

    # * Metrics digits

    Digits = 4

    # * List
    Info = []

    # * Class problem definition
    Class_problem = len(Class_labels)

    if Class_problem == 2:
      Class_problem_prefix = '_Biclass_'
    elif Class_problem > 2:
      Class_problem_prefix = '_Multiclass_'

    # * Training fit

    Start_training_time = time.time()

    Pretrained_model, Pretrained_model_name, Pretrained_model_name_letters = Pretrained_model_function(X_size, Y_size, Class_problem)
    Pretrained_Model_History = Pretrained_model.fit(X_train, y_train, batch_size = 32, validation_split = Vali_split, epochs = Epochs)
  
    End_training_time = time.time()

    
    # * Test evaluation

    Start_testing_time = time.time()

    Loss_Test, Accuracy_Test = Pretrained_model.evaluate(X_test, y_test, verbose = 2)

    End_testing_time = time.time()

    
    # * Total time of training and testing

    Total_training_time = End_training_time - Start_training_time 
    Total_testing_time = End_testing_time - Start_testing_time

    Pretrained_model_name_technique = str(Pretrained_model_name_letters) + '_' + str(Enhancement_technique)

    if Class_problem == 2:

      Labels_biclass_number = []

      for i in range(len(Class_labels)):
        Labels_biclass_number.append(i)

      # * Get the data from the model chosen
      y_pred = Pretrained_model.predict(X_test)
      y_pred = Pretrained_model.predict(X_test).ravel()

      # * Biclass labeling
      y_pred_class = np.where(y_pred < 0.5, 0, 1)
      
      # * Confusion Matrix
      print('Confusion Matrix')
      Confusion_matrix = confusion_matrix(y_test, y_pred_class)

      print(Confusion_matrix)
      print(classification_report(y_test, y_pred_class, target_names = Class_labels))

      # * Precision
      Precision = precision_score(y_test, y_pred_class)
      print(f"Precision: {round(Precision, Digits)}")
      print("\n")

      # * Recall
      Recall = recall_score(y_test, y_pred_class)
      print(f"Recall: {round(Recall, Digits)}")
      print("\n")

      # * F1-score
      F1_score = f1_score(y_test, y_pred_class)
      print(f"F1: {round(F1_score, Digits)}")
      print("\n")

      #print(y_pred_class)
      #print(y_test)

      #print('Confusion Matrix')
      #ConfusionM_Multiclass = confusion_matrix(y_test, y_pred_class)
      #print(ConfusionM_Multiclass)

      #Labels = ['Benign_W_C', 'Malignant']
      Confusion_matrix_dataframe = pd.DataFrame(Confusion_matrix, range(len(Confusion_matrix)), range(len(Confusion_matrix[0])))

      # * Figure's size
      plt.figure(figsize = (Width, Height))
      plt.subplot(X_size_figure, Y_size_figure, 4)
      sns.set(font_scale = font)

      # * Confusion matrix heatmap
      ax = sns.heatmap(Confusion_matrix_dataframe, annot = True, fmt = 'd')
      #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
      ax.set_xlabel('\nPredicted Values')
      ax.set_ylabel('Actual Values')
      ax.set_xticklabels(Class_labels)
      ax.set_yticklabels(Class_labels)

      Accuracy = Pretrained_Model_History.history['accuracy']
      Validation_accuracy = Pretrained_Model_History.history['val_accuracy']

      Loss = Pretrained_Model_History.history['loss']
      Validation_loss = Pretrained_Model_History.history['val_loss']

      # * FPR and TPR values for the ROC curve
      FPR, TPR, _ = roc_curve(y_test, y_pred)
      Auc = auc(FPR, TPR)

      # * Subplot Training accuracy
      plt.subplot(X_size_figure, Y_size_figure, 1)
      plt.plot(Accuracy, label = 'Training Accuracy')
      plt.plot(Validation_accuracy, label = 'Validation Accuracy')
      plt.ylim([0, 1])
      plt.legend(loc = 'lower right')
      plt.title('Training and Validation Accuracy')
      plt.xlabel('epoch')

      # * Subplot Training loss
      plt.subplot(X_size_figure, Y_size_figure, 2)
      plt.plot(Loss, label = 'Training Loss')
      plt.plot(Validation_loss, label = 'Validation Loss')
      plt.ylim([0, 2.0])
      plt.legend(loc = 'upper right')
      plt.title('Training and Validation Loss')
      plt.xlabel('epoch')

      # * Subplot ROC curve
      plt.subplot(X_size_figure, Y_size_figure, 3)
      plt.plot([0, 1], [0, 1], 'k--')
      plt.plot(FPR, TPR, label = Pretrained_model_name + '(area = {:.4f})'.format(Auc))
      plt.xlabel('False positive rate')
      plt.ylabel('True positive rate')
      plt.title('ROC curve')
      plt.legend(loc = 'lower right')

      # * Save this figure in the folder given
      Class_problem_name = str(Class_problem_prefix) + str(Pretrained_model_name) + str(Enhancement_technique) + '.png'
      Class_problem_folder = os.path.join(Folder_models, Class_problem_name)

      plt.savefig(Class_problem_folder)
      #plt.show()

    elif Class_problem >= 3:
    
      Labels_multiclass_number = []

      for i in range(len(Class_labels)):
        Labels_multiclass_number.append(i)

      # * Get the data from the model chosen
      y_pred = Pretrained_model.predict(X_test)
      y_pred = np.argmax(y_pred, axis = 1)

      # * Multiclass labeling
      y_pred_roc = label_binarize(y_pred, classes = Labels_multiclass_number)
      y_test_roc = label_binarize(y_test, classes = Labels_multiclass_number)

      #print(y_pred)
      #print(y_test)

      # * Confusion Matrix
      print('Confusion Matrix')
      Confusion_matrix = confusion_matrix(y_test, y_pred)

      print(Confusion_matrix)
      print(classification_report(y_test, y_pred, target_names = Class_labels))

      # * Precision
      Precision = precision_score(y_test, y_pred, average = 'weighted')
      print(f"Precision: {round(Precision, Digits)}")
      print("\n")

      # * Recall
      Recall = recall_score(y_test, y_pred, average = 'weighted')
      print(f"Recall: {round(Recall, Digits)}")
      print("\n")

      # * F1-score
      F1_score = f1_score(y_test, y_pred, average = 'weighted')
      print(f"F1: {round(F1_score, Digits)}")
      print("\n")

      #labels = ['Benign', 'Benign_W_C', 'Malignant']
      df_cm = pd.DataFrame(Confusion_matrix, range(len(Confusion_matrix)), range(len(Confusion_matrix[0])))

      plt.figure(figsize = (Width, Height))
      plt.subplot(X_size_figure, Y_size_figure, 4)
      sns.set(font_scale = font) # for label size

      ax = sns.heatmap(df_cm, annot = True, fmt = 'd', annot_kws = {"size": Annot_kws}) # font size
      #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
      ax.set_xlabel('\nPredicted Values')
      ax.set_ylabel('Actual Values ')
      ax.set_xticklabels(Class_labels)
      ax.set_yticklabels(Class_labels)

      Accuracy = Pretrained_Model_History.history['accuracy']
      Validation_Accuracy = Pretrained_Model_History.history['val_accuracy']

      Loss = Pretrained_Model_History.history['loss']
      Validation_Loss = Pretrained_Model_History.history['val_loss']

      FPR = dict()
      TPR = dict()
      Roc_auc = dict()

      for i in range(Class_problem):
        FPR[i], TPR[i], _ = roc_curve(y_test_roc[:, i], y_pred_roc[:, i])
        Roc_auc[i] = auc(FPR[i], TPR[i])

      # * Colors for ROC curves
      Colors = cycle(['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan'])
      
      # * Subplot Training accuracy
      plt.subplot(X_size_figure, Y_size_figure, 1)
      plt.plot(Accuracy, label = 'Training Accuracy')
      plt.plot(Validation_Accuracy, label = 'Validation Accuracy')
      plt.ylim([0, 1])
      plt.legend(loc = 'lower right')
      plt.title('Training and Validation Accuracy')
      plt.xlabel('epoch')

      # * Subplot Training loss
      plt.subplot(X_size_figure, Y_size_figure, 2)
      plt.plot(Loss, label = 'Training Loss')
      plt.plot(Validation_Loss, label = 'Validation Loss')
      plt.ylim([0, 2.0])
      plt.legend(loc = 'upper right')
      plt.title('Training and Validation Loss')
      plt.xlabel('epoch')

      # * Subplot ROC curves
      plt.subplot(X_size_figure, Y_size_figure, 3)
      plt.plot([0, 1], [0, 1], 'k--')

      for i, color, lbl in zip(range(Class_problem), Colors, Class_labels):
        plt.plot(FPR[i], TPR[i], color = color, label = 'ROC Curve of class {0} (area = {1:0.4f})'.format(lbl, Roc_auc[i]))

      plt.legend(loc = 'lower right')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False positive rate')
      plt.ylabel('True positive rate')
      plt.title('ROC curve Multiclass')

      # * Save this figure in the folder given
      Class_problem_name = str(Class_problem_prefix) + str(Pretrained_model_name) + str(Enhancement_technique) + '.png'
      Class_problem_folder = os.path.join(Folder_models, Class_problem_name)

      plt.savefig(Class_problem_folder)
      #plt.show()

    Info.append(Pretrained_model_name_technique)
    Info.append(Pretrained_model_name)
    Info.append(Accuracy[Epochs - 1])
    Info.append(Accuracy[0])
    Info.append(Accuracy_Test)
    Info.append(Loss[Epochs - 1])
    Info.append(Loss_Test)
    Info.append(len(y_train))
    Info.append(len(y_test))
    Info.append(Precision)
    Info.append(Recall)
    Info.append(F1_score)
    Info.append(Total_training_time)
    Info.append(Total_testing_time)
    Info.append(Enhancement_technique)
    Info.append(Confusion_matrix[0][0])
    Info.append(Confusion_matrix[0][1])
    Info.append(Confusion_matrix[1][0])
    Info.append(Confusion_matrix[1][1])
    Info.append(Epochs)
  
    if Class_problem == 2:
      Info.append(Auc)
    elif Class_problem > 2:
      for i in range(Class_problem):
        Info.append(Roc_auc[i])
    
    return Info

# ? Update CSV changing value

def overwrite_row_CSV(Dataframe, Folder_path, Info_list, Column_names, Row):

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

    return Dataframe


# ? Folder Configuration of each DCNN model

def configuration_models_folder(Training_data, Validation_data, Test_data, Dataframe_save, Folder_path, DL_model, Enhancement_technique, Class_labels, Column_names, X_size, Y_size, Vali_split, Epochs, Folder_models, Folder_models_esp):

    for Index, Model in enumerate(DL_model):

      Info_model = deep_learning_models_folder(Training_data, Validation_data, Test_data, Model, Enhancement_technique, Class_labels, X_size, Y_size, Vali_split, Epochs, Folder_models, Folder_models_esp)
      
      Info_dataframe = overwrite_row_CSV_folder(Dataframe_save, Folder_path, Info_model, Column_names, Index)

    return Info_dataframe

# ? Folder Pretrained model configurations

def deep_learning_models_folder(Train_data, Valid_data, Test_data, Pretrained_model_function, Enhancement_technique, Class_labels, X_size, Y_size, Vali_split, Epochs, Folder_models, Folder_models_Esp):

    """
	  General configuration for each model, extracting features and printing theirs values.

    Parameters:
    argument1 (model): Model chosen.
    argument2 (str): technique used.
    argument3 (list): labels used for printing.
    argument4 (int): Size of X.
    argument5 (int): Size of Y.
    argument6 (int): Number of classes.
    argument7 (float): Validation split value.
    argument8 (int): Number of epochs.
    argument9 (int): X train split data.
    argument9 (int): y train split data.
    argument9 (int): X test split data.
    argument9 (int): y test split data.
    argument9 (int): Folder used to save data images.
    argument9 (int): Folder used to save data images in spanish.

    Returns:
	  int:Returning all metadata from each model.
    
   	"""

    # * Parameters plt

    batch_size = 32
    Height = 12
    Width = 12
    Annot_kws = 12
    font = 0.7

    X_size_figure = 2
    Y_size_figure = 2

    # * Metrics digits

    Digits = 4

    # * List
    Info = []

    # * Class problem definition
    Class_problem = len(Class_labels)

    if Class_problem == 2:
      Class_problem_prefix = '_Biclass_'
    elif Class_problem > 2:
      Class_problem_prefix = '_Multiclass_'

    # * Training fit

    Start_training_time = time.time()

    Pretrained_model, Pretrained_model_name, Pretrained_model_name_letters = Pretrained_model_function(X_size, Y_size, Class_problem)

    Pretrained_Model_History = Pretrained_model.fit_generator(  Train_data,
                                                      validation_data = Valid_data,
                                                      steps_per_epoch = Train_data.n//Train_data.batch_size,
                                                      validation_steps = Valid_data.n//Valid_data.batch_size,
                                                      epochs = 8)
  
    End_training_time = time.time()

    
    # * Test evaluation

    Start_testing_time = time.time()

    Loss_Test, Accuracy_Test = Pretrained_model.evaluate_generator(Test_data)

    End_testing_time = time.time()

    
    # * Total time of training and testing

    Total_training_time = End_training_time - Start_training_time 
    Total_testing_time = End_testing_time - Start_testing_time

    Pretrained_model_name_technique = str(Pretrained_model_name_letters) + '_' + str(Enhancement_technique)

    if Class_problem == 2:

      Labels_biclass_number = []

      for i in range(len(Class_labels)):
        Labels_biclass_number.append(i)

      # * Get the data from the model chosen

      Predict = Pretrained_model.predict_generator(Test_data)
      y_classes = Predict.argmax(axis = -1)

      y_pred = Pretrained_model.predict(X_test)
      y_pred = Pretrained_model.predict(X_test).ravel()

      # * Biclass labeling
      y_pred_class = np.where(y_pred < 0.5, 0, 1)
      
      # * Confusion Matrix
      print('Confusion Matrix')
      Confusion_matrix = confusion_matrix(y_test, y_pred_class)

      print(Confusion_matrix)
      print(classification_report(y_test, y_pred_class, target_names = Class_labels))

      # * Precision
      Precision = precision_score(y_test, y_pred_class)
      print(f"Precision: {round(Precision, Digits)}")
      print("\n")

      # * Recall
      Recall = recall_score(y_test, y_pred_class)
      print(f"Recall: {round(Recall, Digits)}")
      print("\n")

      # * F1-score
      F1_score = f1_score(y_test, y_pred_class)
      print(f"F1: {round(F1_score, Digits)}")
      print("\n")

      #print(y_pred_class)
      #print(y_test)

      #print('Confusion Matrix')
      #ConfusionM_Multiclass = confusion_matrix(y_test, y_pred_class)
      #print(ConfusionM_Multiclass)

      #Labels = ['Benign_W_C', 'Malignant']
      Confusion_matrix_dataframe = pd.DataFrame(Confusion_matrix, range(len(Confusion_matrix)), range(len(Confusion_matrix[0])))

      # * Figure's size
      plt.figure(figsize = (Width, Height))
      plt.subplot(X_size_figure, Y_size_figure, 4)
      sns.set(font_scale = font)

      # * Confusion matrix heatmap
      ax = sns.heatmap(Confusion_matrix_dataframe, annot = True, fmt = 'd')
      #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
      ax.set_xlabel('\nPredicted Values')
      ax.set_ylabel('Actual Values')
      ax.set_xticklabels(Class_labels)
      ax.set_yticklabels(Class_labels)

      Accuracy = Pretrained_Model_History.history['accuracy']
      Validation_accuracy = Pretrained_Model_History.history['val_accuracy']

      Loss = Pretrained_Model_History.history['loss']
      Validation_loss = Pretrained_Model_History.history['val_loss']

      # * FPR and TPR values for the ROC curve
      FPR, TPR, _ = roc_curve(y_test, y_pred)
      Auc = auc(FPR, TPR)

      # * Subplot Training accuracy
      plt.subplot(X_size_figure, Y_size_figure, 1)
      plt.plot(Accuracy, label = 'Training Accuracy')
      plt.plot(Validation_accuracy, label = 'Validation Accuracy')
      plt.ylim([0, 1])
      plt.legend(loc = 'lower right')
      plt.title('Training and Validation Accuracy')
      plt.xlabel('epoch')

      # * Subplot Training loss
      plt.subplot(X_size_figure, Y_size_figure, 2)
      plt.plot(Loss, label = 'Training Loss')
      plt.plot(Validation_loss, label = 'Validation Loss')
      plt.ylim([0, 2.0])
      plt.legend(loc = 'upper right')
      plt.title('Training and Validation Loss')
      plt.xlabel('epoch')

      # * Subplot ROC curve
      plt.subplot(X_size_figure, Y_size_figure, 3)
      plt.plot([0, 1], [0, 1], 'k--')
      plt.plot(FPR, TPR, label = Pretrained_model_name + '(area = {:.4f})'.format(Auc))
      plt.xlabel('False positive rate')
      plt.ylabel('True positive rate')
      plt.title('ROC curve')
      plt.legend(loc = 'lower right')

      # * Save this figure in the folder given
      Class_problem_name = str(Class_problem_prefix) + str(Pretrained_model_name) + str(Enhancement_technique) + '.png'
      Class_problem_folder = os.path.join(Folder_models, Class_problem_name)

      plt.savefig(Class_problem_folder)
      #plt.show()

    elif Class_problem >= 3:
    
      Labels_multiclass_number = []

      for i in range(len(Class_labels)):
        Labels_multiclass_number.append(i)

      # * Get the data from the model chosen
      Predict = Pretrained_model.predict_generator(Test_data)
      y_classes = Predict.argmax(axis = -1)

      # * Multiclass labeling
      y_pred_roc = label_binarize(y_pred, classes = Labels_multiclass_number)
      y_test_roc = label_binarize(y_test, classes = Labels_multiclass_number)

      #print(y_pred)
      #print(y_test)

      # * Confusion Matrix
      print('Confusion Matrix')
      Confusion_matrix = confusion_matrix(y_test, y_pred)

      print(Confusion_matrix)
      print(classification_report(y_test, y_pred, target_names = Class_labels))

      # * Precision
      Precision = precision_score(y_test, y_pred, average = 'weighted')
      print(f"Precision: {round(Precision, Digits)}")
      print("\n")

      # * Recall
      Recall = recall_score(y_test, y_pred, average = 'weighted')
      print(f"Recall: {round(Recall, Digits)}")
      print("\n")

      # * F1-score
      F1_score = f1_score(y_test, y_pred, average = 'weighted')
      print(f"F1: {round(F1_score, Digits)}")
      print("\n")

      #labels = ['Benign', 'Benign_W_C', 'Malignant']
      df_cm = pd.DataFrame(Confusion_matrix, range(len(Confusion_matrix)), range(len(Confusion_matrix[0])))

      plt.figure(figsize = (Width, Height))
      plt.subplot(X_size_figure, Y_size_figure, 4)
      sns.set(font_scale = font) # for label size

      ax = sns.heatmap(df_cm, annot = True, fmt = 'd', annot_kws = {"size": Annot_kws}) # font size
      #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
      ax.set_xlabel('\nPredicted Values')
      ax.set_ylabel('Actual Values ')
      ax.set_xticklabels(Class_labels)
      ax.set_yticklabels(Class_labels)

      Accuracy = Pretrained_Model_History.history['accuracy']
      Validation_Accuracy = Pretrained_Model_History.history['val_accuracy']

      Loss = Pretrained_Model_History.history['loss']
      Validation_Loss = Pretrained_Model_History.history['val_loss']

      FPR = dict()
      TPR = dict()
      Roc_auc = dict()

      for i in range(Class_problem):
        FPR[i], TPR[i], _ = roc_curve(y_test_roc[:, i], y_pred_roc[:, i])
        Roc_auc[i] = auc(FPR[i], TPR[i])

      # * Colors for ROC curves
      Colors = cycle(['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan'])
      
      # * Subplot Training accuracy
      plt.subplot(X_size_figure, Y_size_figure, 1)
      plt.plot(Accuracy, label = 'Training Accuracy')
      plt.plot(Validation_Accuracy, label = 'Validation Accuracy')
      plt.ylim([0, 1])
      plt.legend(loc = 'lower right')
      plt.title('Training and Validation Accuracy')
      plt.xlabel('epoch')

      # * Subplot Training loss
      plt.subplot(X_size_figure, Y_size_figure, 2)
      plt.plot(Loss, label = 'Training Loss')
      plt.plot(Validation_Loss, label = 'Validation Loss')
      plt.ylim([0, 2.0])
      plt.legend(loc = 'upper right')
      plt.title('Training and Validation Loss')
      plt.xlabel('epoch')

      # * Subplot ROC curves
      plt.subplot(X_size_figure, Y_size_figure, 3)
      plt.plot([0, 1], [0, 1], 'k--')

      for i, color, lbl in zip(range(Class_problem), Colors, Class_labels):
        plt.plot(FPR[i], TPR[i], color = color, label = 'ROC Curve of class {0} (area = {1:0.4f})'.format(lbl, Roc_auc[i]))

      plt.legend(loc = 'lower right')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False positive rate')
      plt.ylabel('True positive rate')
      plt.title('ROC curve Multiclass')

      # * Save this figure in the folder given
      Class_problem_name = str(Class_problem_prefix) + str(Pretrained_model_name) + str(Enhancement_technique) + '.png'
      Class_problem_folder = os.path.join(Folder_models, Class_problem_name)

      plt.savefig(Class_problem_folder)
      #plt.show()

    Info.append(Pretrained_model_name_technique)
    Info.append(Pretrained_model_name)
    Info.append(Accuracy[Epochs - 1])
    Info.append(Accuracy[0])
    Info.append(Accuracy_Test)
    Info.append(Loss[Epochs - 1])
    Info.append(Loss_Test)
    Info.append(len(y_train))
    Info.append(len(y_test))
    Info.append(Precision)
    Info.append(Recall)
    Info.append(F1_score)
    Info.append(Total_training_time)
    Info.append(Total_testing_time)
    Info.append(Enhancement_technique)
    Info.append(Confusion_matrix[0][0])
    Info.append(Confusion_matrix[0][1])
    Info.append(Confusion_matrix[1][0])
    Info.append(Confusion_matrix[1][1])
    Info.append(Epochs)
  
    if Class_problem == 2:
      Info.append(Auc)
    elif Class_problem > 2:
      for i in range(Class_problem):
        Info.append(Roc_auc[i])
    
    return Info

# ? Folder Update CSV changing value

def overwrite_row_CSV_folder(Dataframe, Folder_path, Info_list, Column_names, Row):

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

    return Dataframe

# ? Fine-Tuning MLP

def MLPClassificadorTL(x, units, activation):

    """
	  Fine tuning configuration using only MLP.

    Parameters:
    argument1 (list): Layers.
    argument2 (int): The number of units for last layer.
    argument3 (str): Activation used.

    Returns:
	  int:Returning dataframe with all data.
    
   	"""

    x = Flatten()(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(units, activation = activation)(x)

    return x
  
# ? ResNet50

def ResNet50_PreTrained(Xsize, Ysize, num_classes):
  
    """
	  ResNet50 configuration.

    Parameters:
    argument1 (int): X's size value.
    argument2 (int): Y's size value.
    argument3 (int): Number total of classes.

    Returns:
	  int:Returning ResNet50 model.
    int:Returning ResNet50 Name.
    
   	"""

    Model_name = 'ResNet50_Model'
    Model_name_letters = 'RN50'

    ResNet50_Model = ResNet50(input_shape = (Xsize, Ysize, 3), 
                              include_top = False, 
                              weights = "imagenet")

    for layer in ResNet50_Model.layers:
      layer.trainable = False

    if num_classes == 2:
      activation = 'sigmoid'
      loss = "binary_crossentropy"
      units = 1
    else:
      activation = 'softmax'
      units = num_classes
      loss = "SparseCategoricalCrossentropy"

    x = MLPClassificadorTL(ResNet50_Model.output, units, activation)

    ResNet50_model = Model(ResNet50_Model.input, x)

    ResNet50_model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = loss,
        metrics = ['accuracy']
    )

    return ResNet50_model, Model_name, Model_name_letters

def ResNet50V2_PreTrained(Xsize, Ysize, num_classes):

    """
	  ResNet50V2 configuration.

    Parameters:
    argument1 (int): X's size value.
    argument2 (int): Y's size value.
    argument3 (int): Number total of classes.

    Returns:
	  int:Returning ResNet50V2 model.
    int:Returning ResNet50V2 Name.
    
   	"""

    ModelName = 'ResNet50V2_Model'
    ModelNameLetters = 'RN50V2'

    ResNet50V2_Model = ResNet50V2(input_shape = (Xsize, Ysize, 3), 
                                  include_top = False, 
                                  weights = "imagenet")

    for layer in ResNet50V2_Model.layers:
      layer.trainable = False
  
    if num_classes == 2:
      activation = 'sigmoid'
      loss = "binary_crossentropy"
      units = 1
    else:
      activation = 'softmax'
      units = num_classes
      loss = "SparseCategoricalCrossentropy"
      #loss = "KLDivergence"

    x = MLPClassificadorTL(ResNet50V2_Model.output, units, activation)

    ResNet50V2Model = Model(ResNet50V2_Model.input, x)

    ResNet50V2Model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = loss,
        metrics = ['accuracy']
    )

    return ResNet50V2Model, ModelName, ModelNameLetters

def ResNet152_PreTrained(Xsize, Ysize, num_classes):
  
    """
	  ResNet152 configuration.

    Parameters:
    argument1 (int): X's size value.
    argument2 (int): Y's size value.
    argument3 (int): Number total of classes.

    Returns:
	  int:Returning ResNet152 model.
    int:Returning ResNet152 Name.
    
   	"""

    ModelName = 'ResNet152_Model'
    ModelNameLetters = 'RN152'

    ResNet152_Model = ResNet152(input_shape = (Xsize, Ysize, 3), 
                              include_top = False, 
                              weights = "imagenet")

    for layer in ResNet152_Model.layers:
      layer.trainable = False

    if num_classes == 2:
      activation = 'sigmoid'
      loss = "binary_crossentropy"
      units = 1
    else:
      activation = 'softmax'
      units = num_classes
      loss = "SparseCategoricalCrossentropy"
      #loss = "KLDivergence"

    x = MLPClassificadorTL(ResNet152_Model.output, units, activation)

    ResNet152Model = Model(ResNet152_Model.input, x)

    ResNet152Model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = loss,
        metrics = ['accuracy']
    )

    return ResNet152Model, ModelName, ModelNameLetters

def ResNet152V2_PreTrained(Xsize, Ysize, num_classes):
  
    """
	  ResNet152V2 configuration.

    Parameters:
    argument1 (int): X's size value.
    argument2 (int): Y's size value.
    argument3 (int): Number total of classes.

    Returns:
	  int:Returning ResNet152V2 model.
    int:Returning ResNet152V2 Name.
    
   	"""
     
    ModelName = 'ResNet152V2_Model'
    ModelNameLetters = 'RN152V2'

    ResNet152V2_Model = ResNet152V2(input_shape = (Xsize, Ysize, 3), 
                                    include_top = False, 
                                    weights = "imagenet")

    for layer in ResNet152V2_Model.layers:
      layer.trainable = False

    if num_classes == 2:
      activation = 'sigmoid'
      loss = "binary_crossentropy"
      units = 1
    else:
      activation = 'softmax'
      units = num_classes
      loss = "SparseCategoricalCrossentropy"
      #loss = "KLDivergence"

    x = MLPClassificadorTL(ResNet152V2_Model.output, units, activation)

    ResNet152V2Model = Model(ResNet152V2_Model.input, x)

    ResNet152V2Model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = loss,
        metrics = ['accuracy']
    )

    return ResNet152V2Model, ModelName, ModelNameLetters

# ? ResNet50 folder

def ResNet50_pretrained_folder(Xsize, Ysize, num_classes):

    Model_name = 'ResNet50_Model'
    Model_name_letters = 'RN50'
    
    conv_base = ResNet50(weights = 'imagenet', include_top = False, input_shape = (Xsize, Ysize, 3))
    model = Sequential()
    model.add(conv_base)

    if num_classes == 2:
      activation = 'sigmoid'
      loss = "binary_crossentropy"
      units = 1
    else:
      activation = 'softmax'
      units = num_classes
      loss = "categorical_crossentropy"

    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(units, activation = activation))

    conv_base.trainable = False

    model.compile(loss = loss, optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), metrics = ['accuracy'])

    return model, Model_name, Model_name_letters

# ? MobileNet

def MobileNet_Pretrained(Xsize, Ysize, num_classes):
  
    """
	  MobileNet configuration.

    Parameters:
    argument1 (int): X's size value.
    argument2 (int): Y's size value.
    argument3 (int): Number total of classes.

    Returns:
	  int:Returning MobileNet model.
    int:Returning MobileNet Name.
    
   	"""

    ModelName = 'MobileNet_Model'
    ModelNameLetters = 'MN'

    MobileNet_Model = MobileNet(input_shape = (Xsize, Ysize, 3), 
                                              include_top = False, 
                                              weights = "imagenet")

    for layer in MobileNet_Model.layers:
      layer.trainable = False

    if num_classes == 2:
      activation = 'sigmoid'
      loss = "binary_crossentropy"
      units = 1
    else:
      activation = 'softmax'
      units = num_classes
      loss = "SparseCategoricalCrossentropy"
      #loss = "KLDivergence"

    x = MLPClassificadorTL(MobileNet_Model.output, units, activation)

    MobileNetModel = Model(MobileNet_Model.input, x)

    MobileNetModel.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = loss,
        metrics = ["accuracy"]
    )

    return MobileNetModel, ModelName, ModelNameLetters

def MobileNetV3Small_Pretrained(Xsize, Ysize, num_classes):

    """
	  MobileNetV3Small configuration.

    Parameters:
    argument1 (int): X's size value.
    argument2 (int): Y's size value.
    argument3 (int): Number total of classes.

    Returns:
	  int:Returning MobileNetV3Small model.
    int:Returning MobileNetV3Small Name.
    
   	"""

    ModelName = 'MobileNetV3Small_Model'
    ModelNameLetters = 'MNV3S'

    MobileNetV3Small_Model = MobileNetV3Small(input_shape = (Xsize, Ysize, 3), 
                                              include_top = False, 
                                              weights = "imagenet")

    for layer in MobileNetV3Small_Model.layers:
      layer.trainable = False

    if num_classes == 2:
      activation = 'sigmoid'
      loss = "binary_crossentropy"
      units = 1
    else:
      activation = 'softmax'
      units = num_classes
      loss = "SparseCategoricalCrossentropy"
      #loss = "KLDivergence"

    x = MLPClassificadorTL(MobileNetV3Small_Model.output, units, activation)

    MobileNetV3SmallModel = Model(MobileNetV3Small_Model.input, x)

    MobileNetV3SmallModel.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = loss,
        metrics = ["accuracy"]
    )

    return MobileNetV3SmallModel, ModelName, ModelNameLetters

def MobileNetV3Large_Pretrained(Xsize, Ysize, num_classes):

    """
	  MobileNetV3Large configuration.

    Parameters:
    argument1 (int): X's size value.
    argument2 (int): Y's size value.
    argument3 (int): Number total of classes.

    Returns:
	  int:Returning MobileNetV3Large model.
    int:Returning MobileNetV3Large Name.
    
   	"""

    ModelName = 'MobileNetV3Large_Model'
    ModelNameLetters = 'MNV3L'

    MobileNetV3Large_Model = MobileNetV3Large(input_shape = (Xsize, Ysize, 3), 
                                              include_top = False, 
                                              weights = "imagenet")

    for layer in MobileNetV3Large_Model.layers:
      layer.trainable = False

    if num_classes == 2:
      activation = 'sigmoid'
      loss = "binary_crossentropy"
      units = 1
    else:
      activation = 'softmax'
      units = num_classes
      loss = "SparseCategoricalCrossentropy"
      #loss = "KLDivergence"
      
    x = MLPClassificadorTL(MobileNetV3Large_Model.output, units, activation)

    MobileNetV3LargeModel = Model(MobileNetV3Large_Model.input, x)

    MobileNetV3LargeModel.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = loss,
        metrics = ["accuracy"]
    )

    return MobileNetV3LargeModel, ModelName, ModelNameLetters

# ? MobileNet folder

def MobileNet_pretrained_folder(Xsize, Ysize, num_classes):

    Model_name = 'MobileNet_Model'
    Model_name_letters = 'MN'
    
    conv_base = MobileNet(weights = 'imagenet', include_top = False, input_shape = (Xsize, Ysize, 3))
    model = Sequential()
    model.add(conv_base)

    if num_classes == 2:
      activation = 'sigmoid'
      loss = "binary_crossentropy"
      units = 1
    else:
      activation = 'softmax'
      units = num_classes
      loss = "categorical_crossentropy"

    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(units, activation = activation))

    conv_base.trainable = False

    model.compile(loss = loss, optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), metrics = ['accuracy'])

    return model, Model_name, Model_name_letters

def MobileNetV3Small_pretrained_folder():
    conv_base = MobileNetV3Small(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))
    model = Sequential()
    model.add(conv_base)

    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    conv_base.trainable = False
    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ['accuracy'])
    return model

# Xception

def Xception_Pretrained(Xsize, Ysize, num_classes):

    """
	  Xception configuration.

    Parameters:
    argument1 (int): X's size value.
    argument2 (int): Y's size value.
    argument3 (int): Number total of classes.

    Returns:
	  int:Returning Xception model.
    int:Returning Xception Name.
    
   	"""

    ModelName = 'Xception_Model'
    ModelNameLetters = 'Xc'

    Xception_Model = Xception(input_shape = (Xsize, Ysize, 3), 
                              include_top = False, 
                              weights = "imagenet")

    for layer in Xception_Model.layers:
      layer.trainable = False

    if num_classes == 2:
      activation = 'sigmoid'
      loss = "binary_crossentropy"
      units = 1
    else:
      activation = 'softmax'
      units = num_classes
      loss = "SparseCategoricalCrossentropy"
      #loss = "KLDivergence"

    x = MLPClassificadorTL(Xception_Model.output, units, activation)

    XceptionModel = Model(Xception_Model.input, x)

    XceptionModel.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = loss,
        metrics = ["accuracy"]
    )

    return XceptionModel, ModelName, ModelNameLetters

# VGG

def VGG16_PreTrained(Xsize, Ysize, num_classes):

    """
	  VGG16 configuration.

    Parameters:
    argument1 (int): X's size value.
    argument2 (int): Y's size value.
    argument3 (int): Number total of classes.

    Returns:
	  int:Returning VGG16 model.
    int:Returning VGG16 Name.
    
   	"""

    ModelName = 'VGG16_Model'
    ModelNameLetters = 'VGG16'

    VGG16_Model = VGG16(input_shape = (Xsize, Ysize, 3), 
                        include_top = False, 
                        weights = "imagenet")

    for layer in VGG16_Model.layers:
      layer.trainable = False

    if num_classes == 2:
      activation = 'sigmoid'
      loss = "binary_crossentropy"
      units = 1
    else:
      activation = 'softmax'
      units = num_classes
      loss = "SparseCategoricalCrossentropy"
      #loss = "KLDivergence"

    x = MLPClassificadorTL(VGG16_Model.output, units, activation)

    VGG16Model = Model(VGG16_Model.input, x)

    VGG16Model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = loss,
        metrics = ["accuracy"]
    )

    return VGG16Model, ModelName, ModelNameLetters

def VGG19_PreTrained(Xsize, Ysize, num_classes):

    """
	  VGG19 configuration.

    Parameters:
    argument1 (int): X's size value.
    argument2 (int): Y's size value.
    argument3 (int): Number total of classes.

    Returns:
	  int:Returning VGG19 model.
    int:Returning VGG19 Name.
    
   	"""

    ModelName = 'VGG19_Model'
    ModelNameLetters = 'VGG19'

    VGG19_Model = VGG19(input_shape = (Xsize, Ysize, 3), 
                        include_top = False, 
                        weights = "imagenet")

    for layer in VGG19_Model.layers:
      layer.trainable = False

    if num_classes == 2:
      activation = 'sigmoid'
      loss = "binary_crossentropy"
      units = 1
    else:
      activation = 'softmax'
      units = num_classes
      loss = "SparseCategoricalCrossentropy"
      #loss = "KLDivergence"

    x = MLPClassificadorTL(VGG19_Model.output, units, activation)
    
    VGG19Model = Model(VGG19_Model.input, x)

    VGG19Model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = loss,
        metrics = ["accuracy"]
    )

    return VGG19Model, ModelName, ModelNameLetters

# InceptionV3

def InceptionV3_PreTrained(Xsize, Ysize, num_classes):

    """
	  InceptionV3 configuration.

    Parameters:
    argument1 (int): X's size value.
    argument2 (int): Y's size value.
    argument3 (int): Number total of classes.

    Returns:
	  int:Returning InceptionV3 model.
    int:Returning InceptionV3 Name.
    
   	"""

    ModelName = 'InceptionV3_Model'
    ModelNameLetters = 'IV3'

    InceptionV3_Model = InceptionV3(input_shape = (Xsize, Ysize, 3), 
                                    include_top = False, 
                                    weights = "imagenet")

    for layer in InceptionV3_Model.layers:
      layer.trainable = False

    if num_classes == 2:
      activation = 'sigmoid'
      loss = "binary_crossentropy"
      units = 1
    else:
      activation = 'softmax'
      units = num_classes
      loss = "SparseCategoricalCrossentropy"
      #loss = "KLDivergence"

    x = MLPClassificadorTL(InceptionV3_Model.output, units, activation)

    InceptionV3Model = Model(InceptionV3_Model.input, x)

    InceptionV3Model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = loss,
        metrics = ["accuracy"]
    )

    return InceptionV3Model, ModelName, ModelNameLetters

# DenseNet

def DenseNet121_PreTrained(Xsize, Ysize, num_classes):

    """
	  DenseNet121 configuration.

    Parameters:
    argument1 (int): X's size value.
    argument2 (int): Y's size value.
    argument3 (int): Number total of classes.

    Returns:
	  int:Returning DenseNet121 model.
    int:Returning DenseNet121 Name.
    
   	"""

    ModelName = 'DenseNet121_Model'
    ModelNameLetters = 'DN121'

    DenseNet121_Model = DenseNet121(input_shape = (Xsize, Ysize, 3), 
                                    include_top = False, 
                                    weights = "imagenet")

    for layer in DenseNet121_Model.layers:
      layer.trainable = False

    if num_classes == 2:
      activation = 'sigmoid'
      loss = "binary_crossentropy"
      units = 1
    else:
      activation = 'softmax'
      units = num_classes
      loss = "SparseCategoricalCrossentropy"
      #loss = "KLDivergence"

    x = MLPClassificadorTL(DenseNet121_Model.output, units, activation)

    DenseNet121Model = Model(DenseNet121_Model.input, x)

    DenseNet121Model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = loss,
        metrics = ["accuracy"]
    )

    return DenseNet121Model, ModelName, ModelNameLetters

def DenseNet201_PreTrained(Xsize, Ysize, num_classes):

    """
	  DenseNet201 configuration.

    Parameters:
    argument1 (int): X's size value.
    argument2 (int): Y's size value.
    argument3 (int): Number total of classes.

    Returns:
	  int:Returning DenseNet201 model.
    int:Returning DenseNet201 Name.
    
   	"""

    ModelName = 'DenseNet201_Model'
    ModelNameLetters = 'DN201'

    DenseNet201_Model = DenseNet201(input_shape = (Xsize, Ysize, 3), 
                                    include_top = False, 
                                    weights = "imagenet")

    for layer in DenseNet201_Model.layers:
      layer.trainable = False

    if num_classes == 2:
      activation = 'sigmoid'
      loss = "binary_crossentropy"
      units = 1
    else:
      activation = 'softmax'
      units = num_classes
      loss = "SparseCategoricalCrossentropy"
      #loss = "KLDivergence"

    x = MLPClassificadorTL(DenseNet201_Model.output, units, activation)

    DenseNet201Model = Model(DenseNet201_Model.input, x)

    DenseNet201Model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = loss,
        metrics = ["accuracy"]
    )

    return DenseNet201Model, ModelName, ModelNameLetters

# Custom AlexNet12

def CustomCNNAlexNet12_Model(Xsize, Ysize, num_classes):

    """
	  Custom AlexNet12 configuration.

    Parameters:
    argument1 (int): X's size value.
    argument2 (int): Y's size value.
    argument3 (int): Number total of classes.

    Returns:
	  int:Returning Custom AlexNet12 model.
    int:Returning Custom AlexNet12 Name.
    
   	"""

    ModelName = 'CustomAlexNet12_Model'
    ModelNameLetters = 'CAN12'

    if num_classes == 2:
      activation = 'sigmoid'
      loss = "binary_crossentropy"
      units = 1
    else:
      activation = 'softmax'
      units = num_classes
      loss = "SparseCategoricalCrossentropy"
      #loss = "KLDivergence"

    CustomCNN_Model = Input(shape = (Xsize, Ysize, 3))

    x = CustomCNN_Model
   
    x = Conv2D(filters = 96, kernel_size = (11, 11), strides = (4,4), activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(x)   
    
    x = Conv2D(filters = 256, kernel_size = (5, 5), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2))(x)

    x = Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3, 3), strides=(2, 2))(x)

    x = Flatten()(x) 

    x = Dense(4096, activation = 'relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(4096, activation = 'relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(units, activation = activation)(x)

    CustomLeNet5Model = Model(CustomCNN_Model, x)
    
    CustomLeNet5Model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = loss,
        metrics = ["accuracy"]
    )

    return CustomLeNet5Model, ModelName, ModelNameLetters

def CustomCNNAlexNet12Tunner_Model(Xsize, Ysize, num_classes, hp):

    """
	  Custom AlexNet12 configuration.

    Parameters:
    argument1 (int): X's size value.
    argument2 (int): Y's size value.
    argument3 (int): Number total of classes.

    Returns:
	  int:Returning Custom AlexNet12 model.
    int:Returning Custom AlexNet12 Name.
    
   	"""

    ModelName = 'CustomAlexNet12_Model'
    ModelNameLetters = 'CAN12'

    if num_classes == 2:
      activation = 'sigmoid'
      loss = "binary_crossentropy"
      units = 1
    else:
      activation = 'softmax'
      units = num_classes
      loss = "SparseCategoricalCrossentropy"
      #loss = "KLDivergence"

    CustomCNN_Model = Input(shape = (Xsize, Ysize, 3))

    x = CustomCNN_Model
   
    x = Conv2D(filters = 96, kernel_size = (11, 11), strides = (4,4), activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(x)   
    
    x = Conv2D(filters = 256, kernel_size = (5, 5), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2))(x)

    x = Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = "same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3, 3), strides=(2, 2))(x)

    x = Flatten()(x) 

    x = Dense(hp.Choice('units', [32, 64, 256, 512, 1024, 2048, 4096]), activation = 'relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(hp.Choice('units', [32, 64, 256, 512, 1024, 2048, 4096]), activation = 'relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(units, activation = activation)(x)

    CustomLeNet5Model = Model(CustomCNN_Model, x)
    
    CustomLeNet5Model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = loss,
        metrics = ["accuracy"]
    )

    return CustomLeNet5Model, ModelName, ModelNameLetters
