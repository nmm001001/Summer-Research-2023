#!/usr/bin/env python

"""
Computes the similarty and distance matrix between two rasters using multiprocessing

@author Iris Yoon
iris.hr.yoon@gmail.com
"""

import h5py
import numpy as np
import os
import time
import sys
from functools import partial
from itertools import combinations
from itertools import product
from multiprocessing import Pool, RawArray
from scipy.signal import correlate


var_dict = {}
cwd = os.getcwd()

def init_worker(X1, X1_shape, X2, X2_shape, limit_len):
    var_dict['X1'] = X1
    var_dict['X1_shape'] = X1_shape
    var_dict['X2'] = X2
    var_dict['X2_shape'] = X2_shape
    var_dict['limit_len'] = limit_len

def pair_similarity(neuron1, neuron2):  
    # Computes similarity between neuron1 of raster1 and neuron2 of raster2.
    # Raster must be a numpy array of shape (n_neurons, n_bins).

    # read raster
    X1_raster = np.frombuffer(var_dict['X1']).reshape(var_dict['X1_shape'])
    X2_raster = np.frombuffer(var_dict['X2']).reshape(var_dict['X2_shape'])
    limit_len = var_dict['limit_len']

    n_bins = X1_raster.shape[1]  

    spikes1 = X1_raster[neuron1,:]
    spikes2 = X2_raster[neuron2,:]
    norm_factor = np.sqrt(np.dot(spikes1, spikes1) * np.dot(spikes2, spikes2))
    correlation = correlate(spikes1, spikes2, mode = 'same')
    score =sum(correlation[n_bins//2-limit_len:n_bins//2+limit_len])/norm_factor
    return score

def main():
    """Computes similarity and distances between two rasters. The script should be called as following:
    `python3 cross_similarity_multiprocessing.py path_to_V1_raster path_to_second_raster limit_len output_path`    
    """
    
    V1_path = sys.argv[1]
    downstream_path = sys.argv[2]
    limit_len = int(sys.argv[3])
    output_file = sys.argv[4]
    tik = time.time()
    print("*** This script computes the cross-similarity matrix between two rasters *** ")
 
    ##### Input file path #####
    
    # input path to rasters
    # ex) raster_file = 'multi_system_rasters.h5'
    #V1_path = input("Input path to V1 raster: ")
    #region = input("Input region name (stimulus, orientation, or direction): ")
    #downstream_path = input("Input path to downstream neuron raster: ")
    #limit_len = input('Input limit length: ')
    #limit_len = int(limit_len)
    #output_file = input("Input path to output file. Must end with .h5 :") 
    
    ##### Prepare raster for child process #####
    # open V1 raster
    f = h5py.File(V1_path,'r')
    raster1 = np.array(f['raster'])
    f.close()

    # open raster of selected region
    f = h5py.File(downstream_path,'r')
    raster2 = np.array(f['raster'])
    f.close()

    # Trim V1 raster # this is due to the way to generated the downstream neurons 
    if raster1.shape[1] != raster2.shape[1]:
        diff = raster1.shape[1] - raster2.shape[1]
        raster1 = raster1[:, diff:]

    if raster1.shape[1] != raster2.shape[1]:
        raise AssertionError('Rasters must have equal number of timebins')

    raster1_shape = raster1.shape
    n_neurons1 = raster1.shape[0]
    raster2_shape = raster2.shape
    n_neurons2 = raster2.shape[0]

    # use RawArray, since we only need read access
    X1 = RawArray('d', raster1_shape[0] * raster1_shape[1])
    X2 = RawArray('d', raster2_shape[0] * raster2_shape[1])

    # wrap X as an numpy array
    X1_np = np.frombuffer(X1, dtype = np.float64).reshape(raster1_shape)
    X2_np = np.frombuffer(X2, dtype = np.float64).reshape(raster2_shape)

    # copy data to shared array X
    np.copyto(X1_np, raster1)
    np.copyto(X2_np, raster2)

    ##### using multiprocessor #####
    neuron_comb = list(product(range(n_neurons1), range(n_neurons2)))
    n_processes = 50
    pool = Pool(processes = n_processes, initializer=init_worker, initargs=(X1, raster1_shape, X2, raster2_shape, limit_len))
    data_list = neuron_comb
    output = pool.starmap(pair_similarity, data_list)
    pool.close()

    ##### Compute distance matrix #####
    # create similarity matrix
    similarity = np.zeros((n_neurons1, n_neurons2))
    index = list(product(range(n_neurons1), range(n_neurons2)))
    row_index = np.array([item[0] for item in index])
    col_index = np.array([item[1] for item in index])
    similarity[(row_index, col_index)] = output
    
    # scale and convert to distance matrix 
    similarity_scaled = similarity/np.ceil(np.max(similarity))
    distance = 1-similarity_scaled
    #np.fill_diagonal(distance, 0)   

    ##### Save output #####
    hf = h5py.File(output_file, 'a')
    hf.create_dataset('similarity', data = similarity)
    hf.create_dataset('distance', data = distance)
    hf.close()

    tok = time.time()
    duration = tok-tik
    print('Computation time: %.2f' %duration)

if __name__ == '__main__':
    main()