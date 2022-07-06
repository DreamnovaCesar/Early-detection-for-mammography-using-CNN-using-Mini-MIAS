import pandas as pd

from MIAS_2_Folders import DataModels
from MIAS_4_MIAS_Functions import BarCharModels

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

NameModel = 0
AccuracyFirst = 2  
AccuracyLast = 3
AccuracyTest = 4

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

LossTraining = 5
LossTesting = 6

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Precision = 9
Recall = 10
F1Score = 11

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Timetraining = 12
TimeTest = 13

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

TN = 15
FP = 16
FN = 17
TP = 18

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

AUC = 20

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Folder_Path = pd.read_csv("D:\MIAS\MIAS VS\DataCSV\Biclass_DataFrame_MIAS_Data.csv")
Reverse = True
Biclass = 2

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

#Parameters = [Folder_Path, Title, Label, Data, Reverse]

ParametersTrainingFE = [Folder_Path, "Best accuracy training FE", "Percentage", AccuracyLast, False, DataModels, Biclass]
ParametersTrainingLE = [Folder_Path, "Best accuracy training LE", "Percentage", AccuracyFirst, False, DataModels, Biclass]
ParametersTesting = [Folder_Path, "Best accuracy testing", "Percentage", AccuracyTest, False, DataModels, Biclass]

ParametersLossTraining = [Folder_Path, "Best loss training", "Percentage", LossTraining, True, DataModels, Biclass]
ParametersLossTesting = [Folder_Path, "Best loss testing", "Percentage", LossTesting, True, DataModels, Biclass]

ParametersPrecision = [Folder_Path, "Best precision", "Percentage", Precision, False, DataModels, Biclass]
ParametersRecall = [Folder_Path, "Best recall", "Percentage", Recall, False, DataModels, Biclass]
ParametersF1Score = [Folder_Path, "Best F1score", "Percentage", F1Score, False, DataModels, Biclass]

ParametersTimeTraining = [Folder_Path, "Best time training", "Seconds", Timetraining, True, DataModels, Biclass]
ParametersTimeTesting = [Folder_Path, "Best time testing", "Seconds", TimeTest, True, DataModels, Biclass]

ParametersAUC = [Folder_Path, "Best AUC", "Percentage", AUC, False, DataModels, Biclass]

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

BarCharModels(ParametersTrainingFE)
BarCharModels(ParametersTrainingLE)
BarCharModels(ParametersTesting)

BarCharModels(ParametersLossTraining)
BarCharModels(ParametersLossTesting)

BarCharModels(ParametersPrecision)
BarCharModels(ParametersRecall)
BarCharModels(ParametersF1Score)

BarCharModels(ParametersTimeTraining)
BarCharModels(ParametersTimeTesting)

BarCharModels(ParametersAUC)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########


import pandas as pd

from MIAS_2_Folders import DataModelsML_FO
from MIAS_2_Folders import DataModelsML_SO

from MIAS_4_MIAS_Functions import BarCharModels

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

NameModel = 0
Accuracy = 1  

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Precision = 2
Recall = 3
F1Score = 4

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Timetraining = 7

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

TN = 9
FP = 10
FN = 11
TP = 12

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

AUC = 13

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Folder_Path = pd.read_csv("D:\MIAS\MIAS VS\DataCSV\DataFrame_Binary_MIAS_Data_ML_FO.csv")
Biclass = 2

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

#Parameters = [Folder_Path, Title, Label, Data, Reverse]

ParametersAccuracy = [Folder_Path, "Best accuracy", "Percentage", Accuracy, False, DataModelsML_FO, Biclass]

ParametersPrecision = [Folder_Path, "Best precision", "Percentage", Precision, False, DataModelsML_FO, Biclass]
ParametersRecall = [Folder_Path, "Best recall", "Percentage", Recall, False, DataModelsML_FO, Biclass]
ParametersF1Score = [Folder_Path, "Best F1score", "Percentage", F1Score, True, DataModelsML_FO, Biclass]

ParametersTimetraining = [Folder_Path, "Best time training", "Seconds", Timetraining, True, DataModelsML_FO, Biclass]

ParametersAUC = [Folder_Path, "Best AUC", "Percentage", AUC, False, DataModelsML_FO, Biclass]

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

BarCharModels(ParametersAccuracy)

BarCharModels(ParametersPrecision)
BarCharModels(ParametersRecall)
BarCharModels(ParametersF1Score)

BarCharModels(ParametersTimetraining)

BarCharModels(ParametersAUC)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Folder_Path = pd.read_csv("D:\MIAS\MIAS VS\DataCSV\DataFrame_Binary_MIAS_Data_ML_SO.csv")

ParametersAccuracy = [Folder_Path, "Best accuracy", "Percentage", Accuracy, False, DataModelsML_SO, Biclass]

ParametersPrecision = [Folder_Path, "Best precision", "Percentage", Precision, False, DataModelsML_SO, Biclass]
ParametersRecall = [Folder_Path, "Best recall", "Percentage", Recall, False, DataModelsML_SO, Biclass]
ParametersF1Score = [Folder_Path, "Best F1score", "Percentage", F1Score, True, DataModelsML_SO, Biclass]

ParametersTimetraining = [Folder_Path, "Best time training", "Seconds", Timetraining, True, DataModelsML_SO, Biclass]

ParametersAUC = [Folder_Path, "Best AUC", "Percentage", AUC, False, DataModelsML_SO, Biclass]

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

BarCharModels(ParametersAccuracy)

BarCharModels(ParametersPrecision)
BarCharModels(ParametersRecall)
BarCharModels(ParametersF1Score)

BarCharModels(ParametersTimetraining)

BarCharModels(ParametersAUC)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

