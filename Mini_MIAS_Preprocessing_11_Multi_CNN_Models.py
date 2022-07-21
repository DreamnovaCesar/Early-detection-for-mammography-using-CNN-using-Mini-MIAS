import os
import pandas as pd

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin')

from Mini_MIAS_1_Folders import Multiclass_Data_CSV
from Mini_MIAS_1_Folders import Multiclass_Data_Model
from Mini_MIAS_1_Folders import Multiclass_Data_Model_Esp

from Mini_MIAS_8_CNN_Architectures import configuration_models

def Testing_CNN_Models_Multiclass(Model, Technique, All_images, All_labels):

    # * Parameters
    #Labels_biclass = ['Normal', 'Tumor']
    Labels_triclass = ['Normal', 'Benign', 'Malignant']
    X_size = 224
    Y_size = 224
    Epochs = 5
    Valid_split = 0.1

    # * Lists
    Column_names = ['Name Model', "Model used", "Accuracy Training FE", "Accuracy Training LE", "Accuracy Testing", "Loss Train", "Loss Test", "Training images", "Test images", "Precision", "Recall", "F1_Score", "Time training", "Time testing", "Technique used", "TN", "FP", "FN", "TP", "Epochs", "Auc Normal", "Auc Benign", "Auc Malignant"]
    #Dataframe_keys = ['Model function', 'Technique', 'Labels', 'Xsize', 'Ysize', 'Validation split', 'Epochs', 'Images 1', 'Labels 1', 'Images 2', 'Labels 2']
    
    Dataframe_save_mias = pd.DataFrame(columns = Column_names)
    
    #for Index, model in enumerate(Model):
        #Model_function, Model_name, Model_name_letters = model(X_size, Y_size, len(Labels_biclass))

    # * Save dataframe in the folder given
    #Dataframe_save_mias_name = 'Biclass_' + 'Dataframe_' + 'CNN_' + str(Technique) + '_' + str(Model_name_letters) + '.csv'
    Dataframe_save_mias_name = 'Multiclass_' + 'Dataframe_' + 'CNN_' + str(Technique) + '.csv'
    Dataframe_save_mias_folder = os.path.join(Multiclass_Data_CSV, Dataframe_save_mias_name)

    Dataframe_save_mias.to_csv(Dataframe_save_mias_folder)
    Dataframe_save_mias = pd.read_csv(Dataframe_save_mias_folder)

    print(Dataframe_save_mias_folder)

    Info_model = configuration_models(All_images, All_labels, Dataframe_save_mias, Dataframe_save_mias_folder, Model, Technique, Labels_triclass, Column_names, X_size, Y_size, Valid_split, Epochs, Multiclass_Data_CSV, Multiclass_Data_Model, Multiclass_Data_Model_Esp)
