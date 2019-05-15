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
import pickle
import pydicom as dicom
import tifffile as tiff
import sys


def read_dicom(dcmfile):
    dcm = dicom.read_file(dcmfile, force=True)
    try:
        dcm.decompress()
        data = dcm.pixel_array
    except AttributeError:
        print('DICOM file seems to be incorrect, but try it anyway.')
        dcm.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
        dcm.decompress()
        data = dcm.pixel_array
    try:
        a = dcm.RescaleSlope
    except AttributeError:
        a = 1
    try:
        b = dcm.RescaleIntercept
    except AttributeError:
        b = 0

    return data


def save_hdr(filename, img, dimension=None, gray=True):
    dims = len(img.shape)
    if not (1 < dims <= 3):
        print('Unsupported number of dimensions')
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
            print('Unsupported file format!')
            sys.exit(1)
   
def create_csv(files):
    try:
        data = pd.read_csv('INbreast.csv')
        sort_file = data.sort_values('File Name')
        X = sort_file.iloc[1:, :].values
        patient_names =[]
        file_names = []
        for i in range(0,len(files)):
            temp = files[i].split('.')
            parts = temp[0].split('_')
            file_names.append(parts[0])
            patient_names.append(parts[1])
        data = {'filenames': file_names, 'patients': patient_names}
        df = pd.DataFrame(data, columns=['File Name','PID'])
        sort_file2 = df.sort_values('File Name')
        Y = sort_file2.iloc[1:, :].values
        X = np.array(X)
        Y = np.array(Y)
        for i in range(0, len(Y)):
            X[i,0] = Y[i,1]
        X = pd.DataFrame(X);
        X.to_csv('patients_files.csv', sep='\t')
        print("Successfully created csv containing filenames with corresponding patients")
    except:
        print("failed to create csv containing filenames with corresponding patients")

def filepathprev(input_data_folder):
    path = input_data_folder.split('/')
    pathlength = len(input_data_folder)
    string = ""
    for i in range(0, pathlength-1):
        string.append(path[i] + '/')
    return string

def get_files(input_data_folder):
    files=[]
    for r, d, f in os.walk(input_data_folder):
        for file in f:
            temp = file.split('.')
            if temp[1] == "dcm":
                files.append(file)
    files = sorted(files)
    return files


def rename_files(input_data_folder):
    files=[]
    for i in range(0, len(input_data_folder)):
        temp = input_data_folder[i].split('.')
        parts = temp[0].split('_')
        if parts[4] == 'ML':
            parts[4] = 'MLO'
        string = parts[1] + '_' + parts[3] + '_' + parts[4] + '.png'
        files.append(string)
    return files
 

def saveaspng(input_data_folder, output_data_folder):    
    files = get_files(input_data_folder)
    print(len(files))
    newnames = rename_files(files)
    print(len(newnames))
    for i in range(0, len(files)):
        data = read_dicom(input_data_folder + '/' + files[i])
        save_hdr(output_data_folder+ '/' + newnames[i], data.astype(np.uint16))
        print(str(i)+'/'+str(len(files)))
    create_csv(files)
    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply CLAHE and CMAP')
    parser.add_argument('--input-data-folder', required=True)
    parser.add_argument('--output-data-folder', required=True)
    args = parser.parse_args()
    
    saveaspng( input_data_folder=args.input_data_folder, output_data_folder=args.output_data_folder, )
