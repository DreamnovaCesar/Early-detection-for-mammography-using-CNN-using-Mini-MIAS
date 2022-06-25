from Mini_MIAS_1_Folders import Mini_MIAS_PGM_MOD
from Mini_MIAS_1_Folders import Mini_MIAS_PNG
from Mini_MIAS_1_Folders import Mini_MIAS_TIFF

from Mini_MIAS_2_General_Functions import changeExtension

def preprocessing_ChangeExtension():

    pgm = '.pgm'
    png = '.png'
    tiff = '.tiff'

    PGMtoPNG = changeExtension(folder = Mini_MIAS_PGM_MOD, newfolder = Mini_MIAS_PNG, extension = pgm, newextension = png)
    PGMtoTIFF = changeExtension(folder = Mini_MIAS_PGM_MOD, newfolder = Mini_MIAS_TIFF, extension = pgm, newextension = tiff)

    PGMtoPNG.ChangeExtension()
    PGMtoTIFF.ChangeExtension()