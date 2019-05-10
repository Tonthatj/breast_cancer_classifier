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
import imageio

def save_hdr(filename, img, dimension=None, gray=True):
    dims = len(img.shape)
    if not (1 < dims <= 3):
        eprint('Unsupported number of dimensions')
        sys.exit(1)
    if dims == 3:  # volumetric data is stored slice by slice
        name, ext = os.path.splitext(filename)
        if dimension is None:
            # smallest:
            if min(img.shape) < 20:
                dimension = np.argmin(img.shape)
            # if NxNxM -> M
            elif img.shape[0] == img.shape[1]:
                dimension = 2
            # if NxMxM -> N
            elif img.shape[1] == img.shape[2]:
                dimension = 0
            # if nothing works, use default 0
            else:
                dimension = 0
        for i in range(img.shape[dimension]):
            fname = '{}_{}{}'.format(name, i, ext)
            if dimension == 0:
                save_hdr(fname, img[i, ...])
            if dimension == 1:
                save_hdr(fname, img[:, i, :])
            if dimension == 2:
                save_hdr(fname, img[..., i])
    else:  # 2D data
        if not gray:
            # Transform gray it to RGB
            arr = np.dstack((img, img, img))
        else:
            arr = img
        if filename.endswith('.png'):
            imageio.imsave(filename, arr, format='PNG-FI')
        elif filename.endswith('.tiff'):
            tiff.imsave(filename, arr)
        else:
            eprint('Unsupported file format!')
            sys.exit(1)


def CLAHE(input_data_folder):
    # r=root, d=directories, f = files
    files=[]
    for r, d, f in os.walk(input_data_folder):
        for file in f:
            temp = file.split('.')
            if temp[1] == "png":
                files.append(file)

    Exams=[]
    for i in range(0,len(files)):
        #print(files[i], input_data_folder)
        image = cv2.imread(input_data_folder+'/'+files[i])
        #image = np.array(imageio.imread(input_data_folder+'/'+files[i]))
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(256,256))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
        bgr = np.array(bgr)
        #heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_HOT)
        #heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_OCEAN)
        #heatmap = np.array(heatmap)
        save_hdr(input_data_folder+'/'+files[i], bgr.astype(np.uint16))
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply CLAHE and CMAP')
    parser.add_argument('--input-data-folder', required=True)
    args = parser.parse_args()

   
    CLAHE( input_data_folder=args.input_data_folder, )
