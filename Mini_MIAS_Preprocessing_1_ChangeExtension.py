from Mini_MIAS_1_Folders import ALLpgmMod
from Mini_MIAS_1_Folders import ALLpng
from Mini_MIAS_1_Folders import ALLtiff

from Mini_MIAS_2_General_Functions import changeExtension

def preprocessing_ChangeExtension():

    pgm = '.pgm'
    png = '.png'
    tiff = '.tiff'

    PGMtoPNG = changeExtension(folder = ALLpgmMod, newfolder = ALLpng, extension = pgm, newextension = png)
    PGMtoTIFF = changeExtension(folder = ALLpgmMod, newfolder = ALLtiff, extension = pgm, newextension = tiff)

    PGMtoPNG.ChangeExtension()
    PGMtoTIFF.ChangeExtension()