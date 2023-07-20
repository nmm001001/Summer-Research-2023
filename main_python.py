import Circ_coords_script as cc

D_P = cc.convert_Dx_to_array(cc.import_distance_matrices("distance_dp.h5"))
D_Q = cc.convert_Dx_to_array(cc.import_distance_matrices("distance_dq.h5"))

tc_DP = cc.gen_toroidal_coords(D_P)
tc_DQ = cc.gen_toroidal_coords(D_Q)

cc.plt_tc_neurons(tc_DP,"DP")
cc.plt_tc_neurons(tc_DQ,"DQ")

cc.plt_tc_on_circle(tc_DP, "DP")
cc.plt_tc_on_circle(tc_DQ, "DQ")