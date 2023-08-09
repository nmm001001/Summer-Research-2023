using StatsBase
using Distributions
using Random
using HDF5
using Plots
using StatsPlots

list_num_reg_2_neurons = [50]
num_trials = 1
sparsity_parameter = 1
num_reg_1_neurons = 50
weight_mean = -0.2
weight_stand_dev = 0.4

for fixed_number_output in list_num_reg_2_neurons

    num_reg_2_neurons = fixed_number_output

    mkpath("output_pop_size_$fixed_number_output")

    for trial_index in 1:num_trials

        mkpath("output_pop_size_$fixed_number_output/trial_$trial_index/learning/step_1")

        connection_matrix = zeros(num_reg_2_neurons, num_reg_1_neurons)

        distr = Normal(weight_mean, weight_stand_dev)

        for col_index in axes(connection_matrix)[2]
            num_rows_to_alter = floor(Int, sparsity_parameter*num_reg_2_neurons)
            rows_to_alter = sample(1:num_reg_2_neurons, num_rows_to_alter, replace = false)
            for row_index in rows_to_alter
                weight = rand(distr)
                weight = min(1, weight)
                weight = max(-1, weight)
                connection_matrix[row_index, col_index] = weight
            end
        end

        h5write("output_pop_size_$fixed_number_output/trial_$trial_index/learning/step_1/step_1_connection_matrix.h5", "matrix", connection_matrix)
        heatmap(connection_matrix)
        png("output_pop_size_$fixed_number_output/trial_$trial_index/learning/step_1/step_1_connection_matrix")
    end
end