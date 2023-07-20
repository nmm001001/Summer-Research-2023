#barcode_script.jl

module barcode_script


using WebIO
#WebIO.install_jupyter_nbextension()

using Revise

using HDF5
# HDF5 v 0.12.5 and Plots v1.4.3 work together :-)
using DelimitedFiles
using JLD2
using FileIO
using Printf
using LinearAlgebra
using MultivariateStats
using Statistics
using Dates
using Plots
using CSV

include("aux_functions/julia_utilities.jl")
using .julia_utilities

include("aux_functions/Eirene_var.jl")
using .Eirene_var

include("aux_functions/extension_method.jl")
using .ext

export
    find_n_largest_bars,
    create_folder,
    clean,
    generate_mouse_circle_path,
    generate_DP,
    generate_FAT_ID_matrix,
    generate_DQ,
    compute_DQP_DPQ,
    VR_and_witness_comp,
    barcode_VR_Wit_comp,
    compute_largest_bar,
    save_barcode,
    analogous_bars


# Finds indices of the n largest bars  barcode, which is to be of form Array(Float64, 2)
function find_n_largest_bars(barcode, n)
    total_num_bars = size(barcode)[1]
    true_num_bars = min(n, total_num_bars)
    bar_lengths = zeros(total_num_bars, 2)
    for index in 1:total_num_bars
        bar_length = barcode[index, 2] - barcode[index, 1]
        bar_lengths[index, 1] = index
        bar_lengths[index, 2] = bar_length
    end
    sorted_bar_lengths = bar_lengths[sortperm(bar_lengths[:, 2], rev=true), :] # sorted by the 2nd column
    n_largest_bars = []
    for index in 1:true_num_bars
        bar_index = floor(Int, sorted_bar_lengths[index, 1])
        append!(n_largest_bars, bar_index)
    end
    return n_largest_bars
end

# initialize variables for data simulation

#global num_reg_2_neurons = 51

# GLOBAL DIMENSION VARIABLE
global dim = 1

# GLOBAL DATE TIME FOR NAMING PNG FILES
#global date_now = Dates.format(Dates.now(), "YMMSS")

# response function 
max_rate = 1
slope = -50
response_function = linear_relu_response_function(max_rate, slope)

function create_folder()
    global date_now = Dates.format(Dates.now(), "YMMSS")
    mkpath("Trial folders/$date_now")
    file = open("date.txt", "w")
    write(file, date_now)
    close(file)
end

function clean()
    # general garbage collection function, deletes files that will be repeated and cause exceptions during multiple trials

    if isfile("h5 files/ground_truth.h5")
        rm("h5 files/ground_truth.h5")
    end

    if isfile("h5 files/reg_1_response_matrix.h5")
        rm("h5 files/reg_1_response_matrix.h5")
    end

    if isfile("h5 files/reg_1_distance.h5")
        rm("h5 files/reg_1_distance.h5")
    end

    if isfile("h5 files/distance_dp.h5")
        rm("h5 files/distance_dp.h5")
    end

    if isfile("h5 files/reg_2_response_matrix.h5")
        rm("h5 files/reg_2_response_matrix.h5")
    end

    if isfile("h5 files/conj_rate_distance.h5")
        rm("h5 files/conj_rate_distance.h5")
    end

    if isfile("h5 files/distance_dq.h5")
        rm("h5 files/distance_dq.h5")
    end

    if isfile("date.txt")
        rm("date.txt")
    end

    if isfile("h5 files/D_Q.h5")
        rm("h5 files/D_Q.h5")
    end

    if isfile("h5 files/D_P.h5")
        rm("h5 files/D_P.h5")
    end
    
end


function generate_mouse_circle_path(num_walks=100, num_steps_per_walk=150, step_size_range=0.02)
    # simulate the mouse walking along circle, formatting the existing function into this function allows for easy variable changes
    return generate_random_skipping_circular_walk(num_walks, num_steps_per_walk, step_size_range)
end


function generate_DP(num_neurons=50)
    global num_reg_1_neurons = num_neurons

    reg_1_neurons = generate_uniform_neurons_on_circle(num_reg_1_neurons)
    
    clean()
    
    create_folder()

    h5write("h5 files/ground_truth.h5", "neurons", reg_1_neurons)
    
    path_on_circle = generate_mouse_circle_path(100, 150, 0.02)
    reg_1_neu_resp_matrix = add_normal_random_noise(calculate_neural_response_matrix(reg_1_neurons, path_on_circle, response_function), 0.05, 0.025)
    
    h5write("h5 files/reg_1_response_matrix.h5", "raster", copy(transpose(reg_1_neu_resp_matrix)))

    global reg_1_raster_path = "h5 files/reg_1_response_matrix.h5"
    reg_1_distance_path = "h5 files/reg_1_distance.h5"

    run(`python compute_similarity_multiprocessing.py $reg_1_raster_path 100 $reg_1_distance_path`)
    
    D_P = vector_to_symmetric_matrix(h5read("h5 files/reg_1_distance.h5", "distance"), num_reg_1_neurons)
    h5write("h5 files/distance_dp.h5", "distance", D_P)

    D_P_heatmap = heatmap(D_P)
    
    VR_P = eirene(D_P, record="all", maxdim=dim)
    VR_P_barcode = barcode(VR_P, dim=dim)

    
    png("Trial folders/$date_now/DP")
    

    return D_P, VR_P, reg_1_neu_resp_matrix
end

function generate_FAT_ID_matrix(rows, cols=num_reg_1_neurons)
    global num_reg_2_neurons = rows
    conn_matrix = zeros(rows,cols)

    for col_index in 1:cols
        for row_offset in -10:9
            #row = trunc(Int64, mod(col_index + row_index, row))
            row = mod(col_index + row_offset - 1, rows) + 1
            conn_matrix[row, col_index] = 1.0
        end
    end

    for row_index in 1:rows
        for col_index in 1:cols
            if conn_matrix[row_index, col_index] != 1
                conn_matrix[row_index, col_index] = -1
            end
        end
    end
    heatmap(conn_matrix)
    png("Trial folders/$date_now/connection_matrix")
    return (conn_matrix)
end

function generate_DQ(conn_matrix, reg_1_neu_resp_matrix)
    
    reg_2_response_matrix = add_normal_random_noise(conn_matrix * reg_1_neu_resp_matrix, 0.05, 0.025)

    h5write("h5 files/reg_2_response_matrix.h5", "raster", copy(transpose(reg_2_response_matrix)))
    global reg_2_raster_path = "h5 files/reg_2_response_matrix.h5"
    reg_2_distance_path = "h5 files/reg_2_distance.h5"
    run(`python compute_similarity_multiprocessing.py $reg_2_raster_path 100 $reg_2_distance_path`)

    D_Q = vector_to_symmetric_matrix(h5read("h5 files/reg_2_distance.h5", "distance"), num_reg_2_neurons)
    h5write("h5 files/distance_dq.h5", "distance", D_Q)

    D_Q_heatmap = heatmap(D_Q)

    VR_Q = eirene(D_Q, record="all", maxdim=dim)
    VR_Q_barcode = barcode(VR_Q, dim=dim)

    png("Trial folders/$date_now/DQ")

    return D_Q, VR_Q, reg_2_response_matrix
end

function compute_DQP_DPQ()
    conj_rate_distance_path = "h5 files/conj_rate_distance.h5"
    run(`python cross_similarity_multiprocessing.py $reg_1_raster_path $reg_2_raster_path 100 $conj_rate_distance_path`)
    DQP = h5read("h5 files/conj_rate_distance.h5", "distance")
    DPQ = copy(transpose(DQP))
    
    return DQP, DPQ
end

function VR_and_witness_comp(DP, DQ, DPQ, DQP)
    VRP = eirene(DP, record="all", maxdim=dim)
    VRQ = eirene(DQ, record="all", maxdim=dim)

    WP = compute_Witness_persistence(DPQ, maxdim=dim)
    WQ = compute_Witness_persistence(DQP, maxdim=dim)
    return VRP, VRQ, WP, WQ
end

function barcode_VR_Wit_comp(VRP, VRQ, WP, WQ)
    barcode_VRP = barcode(VRP, dim=dim)
    save_barcode(barcode_VRP, "barcode_VRP")

    barcode_WP = barcode(WP["eirene_output"], dim=dim)
    save_barcode(barcode_WP, "barcode_WP")

    barcode_WQ = barcode(WQ["eirene_output"], dim=dim)
    save_barcode(barcode_WQ, "barcode_WQ")

    barcode_VRQ = barcode(VRQ, dim=dim)
    save_barcode(barcode_VRQ, "barcode_VRQ")

    return barcode_VRP, barcode_VRQ, barcode_WP, barcode_WQ
end

function compute_largest_bar(barcode)
    largest_bar = -1
    largest_span = 0
    for i in 1:size(barcode,1)
        row = barcode[i,:]
        birth = row[1]
        death = row[2]
        span = abs(death - birth)
        if span > largest_span
            largest_bar = i
            largest_span = span
        end
    end
    return largest_bar
end

function save_barcode(barcode, file_name)
    plot_barcode(barcode, xlims=(0,1))

    png("Trial folders/$date_now/$file_name")
end


function analogous_bars(VRP, DP, VRQ, DQ, WP, witness_bar)
    extension_P, extension_Q = ext.run_similarity_analogous(VR_P = VRP, D_P = DP, VR_Q = VRQ, D_Q = DQ, W_PQ = WP, W_PQ_bar = witness_bar, dim=1 )
    plot_analogous_bars(extension_P, extension_Q, xlims=(0,1))
    png("Trial folders/$date_now/analogous_bars")
end


# function execute_trials(iterations, input_neurons, output_neurons, max_output_neurons=0, step_size=0)
#     for 
#         D_P, VR_P, reg_1_neu_resp_matrix = generate_DP(input_neurons)


#     end
# end


# D_P, VR_P, reg_1_neu_resp_matrix = generate_DP(50)
# conn_matrix = generate_FAT_ID_matrix(60)

# D_Q, VR_Q, reg_2_neu_response_matrix = generate_DQ(conn_matrix, reg_1_neu_resp_matrix)

# DQP, DPQ = compute_DQP_DPQ()

# VRP, VRQ, WP, WQ = VR_and_witness_comp(D_P, D_Q, DPQ, DQP)

# barcode_VRP, barcode_VRQ, barcode_WP, barcode_WQ = barcode_VR_Wit_comp(VRP, VRQ, WP, WQ)

# witness_bar = compute_largest_bar(barcode_WP)

# analogous_bars(VR_P, D_P, VR_Q, D_Q, WP, witness_bar)


end #end of module