import cv2 as c #For image Manupliation
import pandas as pd
from itertools import product #For looping over an image}
import numpy as np
from sklearn.preprocessing import QuantileTransformer #Making data to Normal distribution


def loadImage(path:str) -> tuple:
    img = c.imread(path)
    imgRgb = c.cvtColor(img, c.COLOR_BGR2RGB)
    imgArr = np.array(img)
    return (imgRgb, imgArr.shape[0], imgArr.shape[1])

def imgToDataList(height:int, width:int, image) -> list:
    data = []
    i = 0
    for pos in product(range(height), range(width)): #loop over the image
        y = pos[0] # cordinate y
        x = pos[1] # cordinate x
        R,G,B = image[y,x]
        brigthness = 0.299*R + 0.587*G + 0.114*B 
        to_append = [i,(y,x),(R,G,B),brigthness]
        i += 1
        data.append(to_append)
    return data

def loadDataFrame(data: list):
    df = pd.DataFrame(data=data,columns=["pixelNo","position","RGB","intensity"])
    return df

def quantiletransformDataFrame(dataFrame):
    qt = QuantileTransformer(output_distribution='normal')
    array = np.array(dataFrame["intensity"]).reshape(-1,1)
    dataFrame["quantile"] = qt.fit_transform(array)
    return dataFrame

if __name__ == "__main__":
    loadImage()
    imgToDataList()
    loadDataFrame()
    quantiletransformDataFrame()