import os
from multiprocessing import Pool
import argparse
from functools import partial
import scipy.ndimage
import numpy as np
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import os

def CLAHE(input_data_folder):
    # r=root, d=directories, f = files
    files=[]
    for r, d, f in os.walk(input_data_folder):
        for file in f:
            temp = file.split('.')
            if temp[1] = "png":
                files.append(file)

    Exams=[]
    for i in range(0,len(files)):
        #print(files[i], input_data_folder)
        bgr = cv2.imread(input_data_folder+'/'+files[i])
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(256,256))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        bgr = np.array(bgr)
  
        #heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_HOT)
        heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_OCEAN)
        heatmap = np.array(heatmap)
        img = Image.fromarray(heatmap, 'RGB')
        img.save(input_data_folder+'/'+files[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply CLAHE and CMAP')
    parser.add_argument('--input-data-folder', required=True)
    args = parser.parse_args()

   
    CLAHE( input_data_folder=args.input_data_folder, )
