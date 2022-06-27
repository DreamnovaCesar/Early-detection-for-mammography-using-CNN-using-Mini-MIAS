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

# function fine-tuning MLP

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
    
class pretrainedModels:

  def __init__(self, **kwargs):
    
    self.folder = kwargs.get('folder', None)
    self.newfolder = kwargs.get('newfolder', None)
    self.extension = kwargs.get('extension', None)
    self.newextension = kwargs.get('newextension', None)

# Pretrained model configurations

def PreTrainedModels(ModelPreTrained, Technique, labels, Xsize, Ysize, num_classes, vali_split, epochs, X_train, y_train, X_test, y_test, Folder_Save, Folder_Save_Esp):

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

    # Parameters

    Height = 12
    Width = 12
    Annot_kws = 12
    font = 0.7

    # Score list

    Score = []

    if num_classes != len(labels):
      print('The num of classes is not the same than the num of labels,', 'num_classes = ', num_classes, '-----', 'labels = ', len(labels))
      sys.exit(0)

    if num_classes != len(labels):
      print('The num of classes is the same,', 'num_classes = ', num_classes, '-----', 'labels = ', len(labels))

    if num_classes == 2:
      LabelClassName = '_Biclass_'
    elif num_classes > 2:
      LabelClassName = '_Multiclass_'

    Begin_train = time.time()

    # Data pretrained model

    Pretrained_Model, ModelName, ModelNameLetters = ModelPreTrained(Xsize, Ysize, num_classes)
    Pretrained_Model_History = Pretrained_Model.fit(X_train, y_train, batch_size = 64, validation_split = vali_split, epochs = epochs)
  
    End_train = time.time()

    Begin_test = time.time()

    # Test evaluation

    Loss_Test, Accuracy_Test = Pretrained_Model.evaluate(X_test, y_test, verbose = 2)

    End_test = time.time()

    Time_train = End_train - Begin_train 
    Time_test = End_test - Begin_test

    ModelNameTechnique = str(ModelNameLetters) + '_' + str(Technique)

    if num_classes == 2:

      labels_Biclass_Num = []

      for i in range(len(labels)):
        labels_Biclass_Num.append(i)

      y_pred = Pretrained_Model.predict(X_test)
      y_pred = Pretrained_Model.predict(X_test).ravel()
      y_pred_class = np.where(y_pred < 0.5, 0, 1)

      cm = confusion_matrix(y_test, y_pred_class)

      # Precision
      Precision = precision_score(y_test, y_pred_class)
      print(f"Precision: {round(Precision, 4)}")

      print("\n")
      # Recall
      Recall = recall_score(y_test, y_pred_class)
      print(f"Recall: {round(Recall, 4)}")

      print("\n")
      # F1-score
      F1_Score = f1_score(y_test, y_pred_class)
      print(f"F1: {round(F1_Score, 4)}")

      print(y_pred_class)
      print(y_test)

      print('Confusion Matrix')
      ConfusionM_Multiclass = confusion_matrix(y_test, y_pred_class)
      print(ConfusionM_Multiclass)

      print(classification_report(y_test, y_pred_class, target_names = labels))

      #labels = ['Benign_W_C', 'Malignant']
      df_cm = pd.DataFrame(ConfusionM_Multiclass, range(len(ConfusionM_Multiclass)), range(len(ConfusionM_Multiclass[0])))

      plt.figure(figsize = (Width, Height))
      #technique
      plt.subplot(2, 2, 4)
      sns.set(font_scale = font) # for label size

      ax = sns.heatmap(df_cm, annot = True, fmt = 'd') # font size
      #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
      ax.set_xlabel('\nPredicted Values')
      ax.set_ylabel('Actual Values')
      ax.set_xticklabels(labels)
      ax.set_yticklabels(labels)

      Accuracy = Pretrained_Model_History.history['accuracy']
      Validation_Accuracy = Pretrained_Model_History.history['val_accuracy']

      Loss = Pretrained_Model_History.history['loss']
      Validation_Loss = Pretrained_Model_History.history['val_loss']

      fpr, tpr, thresholds = roc_curve(y_test, y_pred)

      Auc = auc(fpr, tpr)

      plt.subplot(2, 2, 1)
      plt.plot(Accuracy, label = 'Training Accuracy')
      plt.plot(Validation_Accuracy, label = 'Validation Accuracy')
      plt.ylim([0, 1])
      plt.legend(loc = 'lower right')
      plt.title('Training and Validation Accuracy')
      plt.xlabel('epoch')

      plt.subplot(2, 2, 2)
      plt.plot(Loss, label = 'Training Loss')
      plt.plot(Validation_Loss, label = 'Validation Loss')
      plt.ylim([0, 2.0])
      plt.legend(loc = 'upper right')
      plt.title('Training and Validation Loss')
      plt.xlabel('epoch')

      plt.subplot(2, 2, 3)
      plt.plot([0, 1], [0, 1], 'k--')
      plt.plot(fpr, tpr, label = ModelName + '(area = {:.4f})'.format(Auc))
      plt.xlabel('False positive rate')
      plt.ylabel('True positive rate')
      plt.title('ROC curve')
      plt.legend(loc = 'lower right')

      dst = ModelName + LabelClassName + Technique + '.png'
      dstPath = os.path.join(Folder_Save, dst)

      plt.savefig(dstPath)

      ############## ############## ############## ############## ############## ############## ##############

      plt.figure(figsize = (Width, Height))
      #technique
      plt.subplot(2, 2, 4)
      sns.set(font_scale = font) # for label size

      ax = sns.heatmap(df_cm, annot = True, fmt = 'd') # font size
      #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
      ax.set_xlabel('\nValores de predicción')
      ax.set_ylabel('Valores actuales')
      ax.set_xticklabels(labels)
      ax.set_yticklabels(labels)

      plt.subplot(2, 2, 1)
      plt.plot(Accuracy, label = 'Exactitud del entrenamiento')
      plt.plot(Validation_Accuracy, label = 'Exactitud de la validación')
      plt.ylim([0, 1])
      plt.legend(loc = 'lower right')
      plt.title('Exactitud del entrenamiento y validación Accuracy')
      plt.xlabel('Epocas')

      plt.subplot(2, 2, 2)
      plt.plot(Loss, label = 'Perdida del entrenamiento')
      plt.plot(Validation_Loss, label = 'Perdida de la validación')
      plt.ylim([0, 2.0])
      plt.legend(loc = 'upper right')
      plt.title('Perdida del entrenamiento y validación Accuracy')
      plt.xlabel('Epocas')

      plt.subplot(2, 2, 3)
      plt.plot([0, 1], [0, 1], 'k--')
      plt.plot(fpr, tpr, label = ModelName + '(area = {:.4f})'.format(Auc))
      plt.xlabel('Tasa de falsos positivos')
      plt.ylabel('Tasa de verdaderos positivos')
      plt.title('Curva ROC')
      plt.legend(loc = 'lower right')

      dst = ModelName + LabelClassName + Technique + '.png'
      dstPath = os.path.join(Folder_Save_Esp, dst)

      plt.savefig(dstPath)

      #plt.show()

    elif num_classes >= 3:
    
      labels_Triclass_Num = []

      for i in range(len(labels)):
        labels_Triclass_Num.append(i)

      # Test y_pred
      y_pred = Pretrained_Model.predict(X_test)
      y_pred = np.argmax(y_pred, axis = 1)

      y_pred_roc = label_binarize(y_pred, classes = labels_Triclass_Num)
      y_test_roc = label_binarize(y_test, classes = labels_Triclass_Num)

      #print(y_pred)
      #print(y_test)

      print('Confusion Matrix')
      ConfusionM_Multiclass = confusion_matrix(y_test, y_pred)
      cm = confusion_matrix(y_test, y_pred)
      print(ConfusionM_Multiclass)

      # Precision
      Precision = precision_score(y_test, y_pred, average = 'weighted')
      print(f"Precision: {round(Precision, 4)}")

      print("\n")
      # Recall
      Recall = recall_score(y_test, y_pred, average = 'weighted')
      print(f"Recall: {round(Recall, 4)}")

      print("\n")
      # F1-score
      F1_Score = f1_score(y_test, y_pred, average = 'weighted')
      print(f"F1: {round(F1_Score, 4)}")

      print(classification_report(y_test, y_pred, target_names = labels))

      #labels = ['Benign', 'Benign_W_C', 'Malignant']
      df_cm = pd.DataFrame(ConfusionM_Multiclass, range(len(ConfusionM_Multiclass)), range(len(ConfusionM_Multiclass[0])))

      plt.figure(figsize = (Width, Height))
      plt.subplot(2, 2, 4)
      sns.set(font_scale = font) # for label size

      ax = sns.heatmap(df_cm, annot = True, fmt = 'd', annot_kws = {"size": Annot_kws}) # font size
      #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
      ax.set_xlabel('\nPredicted Values')
      ax.set_ylabel('Actual Values ')
      ax.set_xticklabels(labels)
      ax.set_yticklabels(labels)

      Accuracy = Pretrained_Model_History.history['accuracy']
      Validation_Accuracy = Pretrained_Model_History.history['val_accuracy']

      Loss = Pretrained_Model_History.history['loss']
      Validation_Loss = Pretrained_Model_History.history['val_loss']

      fpr = dict()
      tpr = dict()
      roc_auc = dict()

      for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_roc[:, i], y_pred_roc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

      colors = cycle(['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan'])

      plt.subplot(2, 2, 1)
      plt.plot(Accuracy, label = 'Training Accuracy')
      plt.plot(Validation_Accuracy, label = 'Validation Accuracy')
      plt.ylim([0, 1])
      plt.legend(loc = 'lower right')
      plt.title('Training and Validation Accuracy')
      plt.xlabel('epoch')

      plt.subplot(2, 2, 2)
      plt.plot(Loss, label = 'Training Loss')
      plt.plot(Validation_Loss, label = 'Validation Loss')
      plt.ylim([0, 2.0])
      plt.legend(loc = 'upper right')
      plt.title('Training and Validation Loss')
      plt.xlabel('epoch')
    
      plt.subplot(2, 2, 3)
      plt.plot([0, 1], [0, 1], 'k--')

      for i, color, lbl in zip(range(num_classes), colors, labels):
        plt.plot(fpr[i], tpr[i], color = color, label = 'ROC Curve of class {0} (area = {1:0.4f})'.format(lbl, roc_auc[i]))

      plt.legend(loc = 'lower right')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False positive rate')
      plt.ylabel('True positive rate')
      plt.title('ROC curve Multiclass')

      dst = ModelName + LabelClassName + Technique + '.png'
      dstPath = os.path.join(Folder_Save, dst)

      plt.savefig(dstPath)

      ############## ############## ############## ############## ############## ############## ##############

      plt.figure(figsize = (Width, Height))
      #technique
      plt.subplot(2, 2, 4)
      sns.set(font_scale = font) # for label size

      ax = sns.heatmap(df_cm, annot = True, fmt = 'd') # font size
      #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
      ax.set_xlabel('\nValores de predicción')
      ax.set_ylabel('Valores actuales')
      ax.set_xticklabels(labels)
      ax.set_yticklabels(labels)

      fpr = dict()
      tpr = dict()
      roc_auc = dict()

      for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_roc[:, i], y_pred_roc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

      colors = cycle(['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan'])

      plt.subplot(2, 2, 1)
      plt.plot(Accuracy, label = 'Exactitud del entrenamiento')
      plt.plot(Validation_Accuracy, label = 'Exactitud de la validación')
      plt.ylim([0, 1])
      plt.legend(loc = 'lower right')
      plt.title('Exactitud del entrenamiento y validación Accuracy')
      plt.xlabel('Epocas')

      plt.subplot(2, 2, 2)
      plt.plot(Loss, label = 'Perdida del entrenamiento')
      plt.plot(Validation_Loss, label = 'Perdida de la validación')
      plt.ylim([0, 2.0])
      plt.legend(loc = 'upper right')
      plt.title('Perdida del entrenamiento y validación Accuracy')
      plt.xlabel('Epocas')
    
      plt.subplot(2, 2, 3)
      plt.plot([0, 1], [0, 1], 'k--')

      for i, color, lbl in zip(range(num_classes), colors, labels):
        plt.plot(fpr[i], tpr[i], color = color, label = 'ROC Curva de clase {0} (area = {1:0.4f})'.format(lbl, roc_auc[i]))

      plt.legend(loc = 'lower right')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('Tasa de falsos positivos')
      plt.ylabel('Tasa de verdaderos positivos')
      plt.title('ROC Multiclase')

      dst = ModelName + LabelClassName + Technique + '.png'
      dstPath = os.path.join(Folder_Save_Esp, dst)

      plt.savefig(dstPath)

      #plt.show()

    Score.append(ModelNameTechnique)
    Score.append(ModelName)
    Score.append(Accuracy[epochs - 1])
    Score.append(Accuracy[0])
    Score.append(Accuracy_Test)
    Score.append(Loss[epochs - 1])
    Score.append(Loss_Test)
    Score.append(len(y_train))
    Score.append(len(y_test))
    Score.append(Precision)
    Score.append(Recall)
    Score.append(F1_Score)
    Score.append(Time_train)
    Score.append(Time_test)
    Score.append(Technique)
    Score.append(cm[0][0])
    Score.append(cm[0][1])
    Score.append(cm[1][0])
    Score.append(cm[1][1])
    Score.append(epochs)
  
    if num_classes == 2:
      Score.append(Auc)
    elif num_classes > 2:
      for i in range(num_classes):
        Score.append(roc_auc[i])
  
    return Score

# Fine-Tuning MLP

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
  
# ResNet50

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

    ModelName = 'ResNet50_Model'
    ModelNameLetters = 'RN50'

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

    ResNet50Model = Model(ResNet50_Model.input, x)

    ResNet50Model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = loss,
        metrics = ['accuracy']
    )

    return ResNet50Model, ModelName, ModelNameLetters

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

# MobileNet

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
