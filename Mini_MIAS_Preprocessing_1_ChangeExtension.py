from Mini_MIAS_1_Folders import Mini_MIAS_PGM_MOD
from Mini_MIAS_1_Folders import Mini_MIAS_PNG
from Mini_MIAS_1_Folders import Mini_MIAS_TIFF

from Mini_MIAS_2_General_Functions import changeFormat

def preprocessing_ChangeFormat(Current_format, New_format):

    # * With this class we change the format of each image for a new one
    Format_change = changeFormat(Folder = Mini_MIAS_PGM_MOD, Newfolder = Mini_MIAS_PNG, Format = Current_format, Newformat = New_format)
    #PGMtoPNG = changeFormat(Folder = Mini_MIAS_PGM_MOD, Newfolder = Mini_MIAS_PNG, Format = pgm, Newformat = png)
    #PGMtoTIFF = changeFormat(Folder = Mini_MIAS_PGM_MOD, Newfolder = Mini_MIAS_TIFF, Format = pgm, Newformat = tiff)

    Format_change.ChangeExtension()
    #PGMtoPNG.ChangeExtension()
    #PGMtoTIFF.ChangeExtension()