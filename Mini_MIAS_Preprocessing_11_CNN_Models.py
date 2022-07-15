import os
import pandas as pd

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin')

from Mini_MIAS_1_Folders import Biclass_Data_CSV
from Mini_MIAS_1_Folders import Biclass_Data_Model
from Mini_MIAS_1_Folders import Biclass_Data_Model_Esp

from Mini_MIAS_1_Folders import Multiclass_Data_CSV
from Mini_MIAS_1_Folders import Multiclass_Data_Model
from Mini_MIAS_1_Folders import Multiclass_Data_Model_Esp

from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Normal
from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Tumor

from Mini_MIAS_8_CNN_Architectures import MobileNetV3Small_Pretrained
from Mini_MIAS_8_CNN_Architectures import MobileNetV3Large_Pretrained
from Mini_MIAS_8_CNN_Architectures import MobileNet_Pretrained

from Mini_MIAS_8_CNN_Architectures import ResNet50_PreTrained
from Mini_MIAS_8_CNN_Architectures import ResNet50V2_PreTrained
from Mini_MIAS_8_CNN_Architectures import ResNet152V2

from Mini_MIAS_8_CNN_Architectures import configuration_models
from Mini_MIAS_8_CNN_Architectures import overwrite_row_CSV

Model_Tested = MobileNetV3Large_Pretrained

#NT = 'NT'
#NO = 'NO'  
#CLAHE = 'CLAHE'
#HE = 'HE'
#UM = 'UM'
#CS = 'CS' 

def Testing_CNN_Models_Biclass(Model, Technique, All_images, All_labels):

    # * Parameters
    Labels_biclass = ['Normal', 'Tumor']
    #Labels_triclass = ['Normal', 'Benign', 'Malignant']
    X_size = 224
    Y_size = 224
    Epochs = 5
    Valid_split = 0.1

    # * Lists
    Column_names = ['Name Model', "Model used", "Accuracy Training FE", "Accuracy Training LE", "Accuracy Testing", "Loss Train", "Loss Test", "Training images", "Test images", "Precision", "Recall", "F1_Score", "Time training", "Time testing", "Technique used", "TN", "FP", "FN", "TP", "Epochs", "Auc"]
    #Dataframe_keys = ['Model function', 'Technique', 'Labels', 'Xsize', 'Ysize', 'Validation split', 'Epochs', 'Images 1', 'Labels 1', 'Images 2', 'Labels 2']
    
    Dataframe_save_mias = pd.DataFrame(columns = Column_names)
    
    #for Index, model in enumerate(Model):
        #Model_function, Model_name, Model_name_letters = model(X_size, Y_size, len(Labels_biclass))

    # * Save dataframe in the folder given
    #Dataframe_save_mias_name = 'Biclass_' + 'Dataframe_' + 'CNN_' + str(Technique) + '_' + str(Model_name_letters) + '.csv'
    Dataframe_save_mias_name = 'Biclass_' + 'Dataframe_' + 'CNN_' + str(Technique) + '.csv'
    Dataframe_save_mias_folder = os.path.join(Biclass_Data_CSV, Dataframe_save_mias_name)

    Dataframe_save_mias.to_csv(Dataframe_save_mias_folder)
    Dataframe_save_mias = pd.read_csv(Dataframe_save_mias_folder)

    print(Dataframe_save_mias_folder)

    Images_Normal = All_images[0]
    Images_Tumor = All_images[1]

    Labels_Normal = All_labels[0]
    Labels_Tumor = All_labels[1]

    #df = pd.read_csv("D:\Mini-MIAS\Mini-MIAS Final\Biclass_Data_CSV\Biclass_DataFrame_MIAS_Data.csv")
    #path = "D:\Mini-MIAS\Mini-MIAS Final\Biclass_Data_CSV\Biclass_DataFrame_MIAS_Data.csv"

    #parameters_model = [Model, technique, Labels_biclass, X_size, Y_size, Valid_split, Epochs, Images_Normal, Labels_Normal, Images_Tumor, Labels_Tumor]
    #configuration_models(All_images, All_labels, Dataframe_save, DL_model, Enhancement_technique, Class_labels, Column_names, X_size, Y_size, Vali_split, Epochs, Folder_path, Folder_models, Folder_models_esp)
    Info_model = configuration_models(All_images, All_labels, Dataframe_save_mias, Dataframe_save_mias_folder, Model, Technique, Labels_biclass, Column_names, X_size, Y_size, Valid_split, Epochs, Biclass_Data_CSV, Biclass_Data_Model, Biclass_Data_Model_Esp)