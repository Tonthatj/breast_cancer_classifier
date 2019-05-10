import os
from multiprocessing import Pool
import argparse
from functools import partial
import scipy.ndimage
import numpy as np
import pandas as pd





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply CLAHE and CMAP')
    parser.add_argument('--input-data-folder', required=True)
    args = parser.parse_args()

   
    CLAHE( input_data_folder=args.input_data_folder, )
