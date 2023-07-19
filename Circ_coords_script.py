#import sklearn
import matplotlib.pyplot as plt
from dreimac import ToroidalCoords, CircularCoords, GeometryExamples, CircleMapUtils
from persim import plot_diagrams
from sklearn.datasets import make_circles
import numpy as np
import h5py
from datetime import datetime

file = open("date.txt")
date_now = file.read()
file.close()

def import_distance_matrices(D_x):
    # takes string as input (file name)
    D_x = (h5py.File(f"h5 files/{D_x}", "r")).get("distance")
    return D_x



def convert_Dx_to_array(D_x):
    D_x = np.array(D_x)
    return D_x


D_P = convert_Dx_to_array(import_distance_matrices("distance_dp.h5"))
D_Q = convert_Dx_to_array(import_distance_matrices("distance_dq.h5"))


def gen_toroidal_coords(D_x, num_landmarks=50, percent=0.1, cohomology_classes=[0], plot_tc=False):
    tc = ToroidalCoords(D_x, num_landmarks, distance_matrix=True)
    if plot_tc:
        plot_diagrams(tc.dgms_)
        plt.savefig(f"Trial folders/{date_now}/{D_x}_tc_neuron_plot.png")
        plt.close()
    
    toroidal_coords = tc.get_coordinates(perc=percent, cocycle_idxs=cohomology_classes, standard_range=False)
    return toroidal_coords

def plt_tc_neurons(toroidal_coords, D_x):
    if D_x.lower() == "dq":
        neu = np.linspace(0,1,len(D_Q))
    elif D_x.lower() == "dp":
        neu = np.linspace(0,1,len(D_P))
    
    plt.scatter(toroidal_coords, neu)
    plt.savefig(f"Trial folders/{date_now}/{D_x}_tc_neurons.png")
    plt.close()
    return

def plt_tc_on_circle(toroidal_coords, file_name):
    xs = []
    ys = []

    for index in range(len(toroidal_coords)):
        x_val = np.cos(2*np.pi*toroidal_coords[index])
        xs.append(x_val)
        y_val = np.sin(2*np.pi*toroidal_coords[index])
        ys.append(y_val)

    plt.scatter(xs,ys)
    plt.savefig(f"Trial folders/{date_now}/{file_name}_tc_circular.png")
    plt.close()
    return


# D_P = convert_Dx_to_array(import_distance_matrices("distance_dp.h5"))
# D_Q = convert_Dx_to_array(import_distance_matrices("distance_dq.h5"))

tc_DP = gen_toroidal_coords(D_P)
tc_DQ = gen_toroidal_coords(D_Q)

print(type(tc_DP))

plt_tc_neurons(tc_DP,"DP")
plt_tc_neurons(tc_DQ,"DQ")

plt_tc_on_circle(tc_DP, "DP")
plt_tc_on_circle(tc_DQ, "DQ")


