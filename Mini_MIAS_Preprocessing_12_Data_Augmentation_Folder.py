"""
import os
import splitfolders

from Mini_MIAS_1_Folders import Mini_MIAS_NT_Cropped_Images_Biclass
from Mini_MIAS_1_Folders import Mini_MIAS_NO_Cropped_Images_Biclass

def split_folders_train_test_val(Folder):

    #Name_dir = os.path.dirname(Folder)
    #Name_base = os.path.basename(Folder)

    Name_base_mod = Folder + '_Split'

    splitfolders.ratio(Folder, output = Name_base_mod, seed = 1337, ratio = (0.8, 0.1, 0.1)) 

    for (root, dirs, files) in os.walk(Name_base_mod, topdown = True):
        print (root)
        print (dirs)
        #print (files)
        print ('--------------------------------')

split_folders_train_test_val(Mini_MIAS_NT_Cropped_Images_Biclass)
"""