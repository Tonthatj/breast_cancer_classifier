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



def filepathprev(input_data_folder):
    path = input_data_folder.split('/')
    pathlength = len(input_data_folder)
    string = ""
    for i in range(0, pathlength-1):
        string = string + path[i] + '/')
    return string

def get_files(input_data_folder):
    files=[]
    for r, d, f in os.walk(input_data_folder):
        for file in f:
            temp = file.split('.')
            if temp[1] == "png":
                files.append(file)
    return files

def get_dict_elems(files):
    elems = []
    try:
        first_parts = files[0].split('_')
        first = first_parts[0]
        count = 1
        for i in range(1, len(files)):
            parts = files[i].split('_')
            if first == parts[0]:
                count +=1
            else:
                elems.append(count)
                count = 1
                first = parts[0]
        elems.append(count)
    except:
        print("No files in path!")
    return elems

def save_pickle(dictionary, path):
    string = path + "exam_list_before_cropping2.pkl"
    pickle.dump(dictionary, open(string, "wb"))
    
def gen_dict(dict_elems, files):
    dictionary = {}
    n = 0
    for i in range(0, len(dict_elems)):
        if(dict_elems[i] >= 4):
            dicts = { 'horizontal_flip': 'NO', 'L-CC': ['0_L_CC'], 'R-CC': ['0_R_CC'], 'L-MLO': ['0_L_MLO'], 'R-MLO': ['0_R_MLO'], }
            lcc, lmlo, rcc, rmlo = 0,0,0,0
            for j in range(0, dict_elems[i]):
                temp = files[n+j].split('.')
                parts = temp[0].split('_')
                if lcc== 0 and parts[1] == "L" and parts[2] == "CC":
                    temp_dict = {'L-CC' : temp}
                    dicts.update(temp_dict)
                if lmlo== 0 and parts[1] == "L" and parts[2] == "MLO":
                    temp_dict = {'L-MLO' : temp}
                    dicts.update(temp_dict)
                if rcc== 0 and parts[1] == "R" and parts[2] == "CC":
                    temp_dict = {'R-CC' : temp}
                    dicts.update(temp_dict)
                if rmlo== 0 and parts[1] == "R" and parts[2] == "MLO":
                    temp_dict = {'R-MLO' : temp}
                    dicts.update(temp_dict)
        if(dict_elems[i] >= 4):
            dictionary.append(dicts)                 
        n = n + dict_elems[i]
    return dictionary

def gen_pickle(input_data_folder):    
    newpath = filepathprev(input_data_folder)
    files = get_files(input_data_folder)
    elem_dict = get_dict_elems(files)
    dictionary = gen_dict(elem_dict, files)
    save_pickle(dictionary, newpath)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply CLAHE and CMAP')
    parser.add_argument('--input-data-folder', required=True)
    args = parser.parse_args()

    gen_pickle( input_data_folder=args.input_data_folder, )

