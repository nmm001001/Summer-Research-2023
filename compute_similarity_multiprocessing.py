#!/usr/bin/env python
"""
Computes the similarty and distance matrix between neurons in a raster using multiprocessing

@author Iris Yoon
iris.hr.yoon@gmail.com
"""
# Make sure current directory is added in the PATH variable
import h5py
import numpy as np
import os
import time
import sys
from functools import partial
from itertools import combinations
from multiprocessing import Pool, RawArray
from scipy.signal import correlate

var_dict = {}
cwd = os.getcwd()

def init_worker(X, X_shape, limit_len):
    var_dict['X'] = X
    var_dict['X_shape'] = X_shape
    var_dict['limit_len'] = limit_len

def pair_similarity(neuron1, neuron2):  
    # Computes similarity between neuron1 and neuron2 in a raster.
    # Raster must be a numpy array of shape (n_neurons, n_bins).

    # read raster
    X_raster = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])
    limit_len = var_dict['limit_len']

    n_bins = X_raster.shape[1]  

    spikes1 = X_raster[neuron1,:]
    spikes2 = X_raster[neuron2,:]
    norm_factor = np.sqrt(np.dot(spikes1, spikes1) * np.dot(spikes2, spikes2))
    correlation = correlate(spikes1, spikes2, mode = 'same')
    score =sum(correlation[n_bins//2-limit_len:n_bins//2+limit_len])/norm_factor
    return score


def scale_compute_distance(similarity, scale_factor):
    distance =  1- similarity / scale_factor
    return distance


def compute_similarity(raster_path, limit_len, output_file):
    tik = time.time()

    ##### Prepare raster for child process #####
    # open raster
    f = h5py.File(raster_path,'r')
    raster = np.array(f['raster'])
    f.close()

    raster_shape = raster.shape
    n_neurons = raster.shape[0]

    # use RawArray, since we only need read access
    X = RawArray('d', raster_shape[0] * raster_shape[1])

    # wrap X as an numpy array
    X_np = np.frombuffer(X, dtype = np.float64).reshape(raster_shape)

    # copy data to shared array X
    np.copyto(X_np, raster)
   
    ##### using multiprocessor #####
    neuron_comb = list(combinations(range(n_neurons),2))
    n_processes = 50
    pool = Pool(processes = n_processes, initializer=init_worker, initargs=(X, raster_shape, limit_len))
    data_list = neuron_comb
    output = pool.starmap(pair_similarity, data_list)
    pool.close()
    
    # compute distance matrix 
    sim_max = max(output)
    distance = scale_compute_distance(output, sim_max)
    

    ##### Save output #####
    # note: One must convert the output into an array form.
    hf = h5py.File(output_file, 'w')
    hf.create_dataset('similarity', data = output)
    hf.create_dataset('distance', data = distance)
    hf.close()
    
    tok = time.time()
    duration = tok-tik
    print('Computation time for %d neurons: %.2f' %(n_neurons, duration))


def main():
    raster_path = sys.argv[1]
    limit_len = int(sys.argv[2])
    output_file = sys.argv[3]
    tik = time.time()    
    
    ##### Input file path #####
    #raster_path = input("Input path to raster: ")
    #limit_len = input('Input limit length: ')
    #limit_len = int(limit_len)
    #output_file = input("Input path to output file. Must end with .h5 :") 
    
    #
    ##### Prepare raster for child process #####
    # open raster
    f = h5py.File(raster_path,'r')
    raster = np.array(f['raster'])
    f.close()

    raster_shape = raster.shape
    n_neurons = raster.shape[0]

    # use RawArray, since we only need read access
    X = RawArray('d', raster_shape[0] * raster_shape[1])

    # wrap X as an numpy array
    X_np = np.frombuffer(X, dtype = np.float64).reshape(raster_shape)

    # copy data to shared array X
    np.copyto(X_np, raster)
   
    ##### using multiprocessor #####
    neuron_comb = list(combinations(range(n_neurons),2))
    n_processes = 50
    pool = Pool(processes = n_processes, initializer=init_worker, initargs=(X, raster_shape, limit_len))
    data_list = neuron_comb
    output = pool.starmap(pair_similarity, data_list)
    pool.close()
    
    # compute distance matrix 
    sim_max = max(output)
    distance = scale_compute_distance(output, sim_max)
    

    ##### Save output #####
    # note: One must convert the output into an array form.
    hf = h5py.File(output_file, 'w')
    hf.create_dataset('similarity', data = output)
    hf.create_dataset('distance', data = distance)
    hf.close()
    
    tok = time.time()
    duration = tok-tik
    print('Computation time for %d neurons: %.2f' %(n_neurons, duration))
    


if __name__ == '__main__':
    main()


