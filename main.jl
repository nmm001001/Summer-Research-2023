include("barcode_script.jl")
using .barcode_script
# ENV["PYTHON"] = "venv/Scripts/python.exe"
# using Pkg
# Pkg.build("PyCall")
# using PyCall

# circ_coords = pyimport("Circ_coords_script")

using HDF5

i


run(`julia julia_init_conn_matrix.jl`)
run(`julia hebbian_learning_script.jl`)

DP, VRP, reg_1_neu_resp_matrix = generate_DP(50)
               
conn_matrix = h5read("output_pop_size_50/trial_1/learning/step_61/step_61_connection_matrix.h5", "matrix")

DQ, VRQ, reg_2_neu_response_matrix = generate_DQ(conn_matrix, reg_1_neu_resp_matrix)

DQP, DPQ = compute_DQP_DPQ()

VRP, VRQ, WP, WQ = VR_and_witness_comp(DP, DQ, DPQ, DQP)

barcode_VRP, barcode_VRQ, barcode_WP, barcode_WQ = barcode_VR_Wit_comp(VRP, VRQ, WP, WQ)

witness_bar = compute_largest_bar(barcode_WP)

analogous_bars(VRP, DP, VRQ, DQ, WP, witness_bar)

run(`python3 Circ_coords_script.py`)

