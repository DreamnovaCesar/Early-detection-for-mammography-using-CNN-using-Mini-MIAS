from Mini_MIAS_1_Folders import Mini_MIAS_PGM_MOD
from Mini_MIAS_1_Folders import Mini_MIAS_PNG
from Mini_MIAS_1_Folders import Mini_MIAS_TIFF

from Mini_MIAS_2_General_Functions import changeFormat

def preprocessing_ChangeFormat():

    # * General parameters

    pgm = '.pgm'
    png = '.png'
    tiff = '.tiff'

    # * With this class we change the format of each image for a new one
    PGMtoPNG = changeFormat(Folder = Mini_MIAS_PGM_MOD, Newfolder = Mini_MIAS_PNG, Format = pgm, Newformat = png)
    PGMtoTIFF = changeFormat(Folder = Mini_MIAS_PGM_MOD, Newfolder = Mini_MIAS_TIFF, Format = pgm, Newformat = tiff)

    PGMtoPNG.ChangeExtension()
    PGMtoTIFF.ChangeExtension()