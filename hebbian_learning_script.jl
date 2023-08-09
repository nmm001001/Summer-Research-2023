# To run this script, one can either run it from the command line after starting a Julia session
# or in VSCode by "Julia: Run file in new process".
# For whatever reason, VSCode's REPL configuration is different from the base system's Julia REPL's
# and that can cause issues.
# In either case, one must make sure that Julia is using the correct Python interpreter .
# To configure this, within Julia, run "ENV["PYTHON"] = <path to python executable with relevant modules installed>"
# and then enter the pkg envirnoment with "]"
# and then "Pkg.build("PyCall")".


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
using StatsPlots

includet("aux_functions/julia_utilities.jl")
using .julia_utilities

########################################################################
########################################################################

list_num_reg_2_neurons = [50]
# CONSTRUCT BIAS VECTOR BELOW

num_trials = 1
num_reg_1_neurons = 50
num_learning_steps = 60
num_walks = 100
num_steps_per_walk = 150
step_size_range = 0.02
max_rate = 1
slope = -25
learning_threshold = 0.0    
learning_step_size = 1.0

# Declare response function
response_function = linear_relu_response_function(max_rate, slope)

# Generate region 1 neurons, uniformly chosen on circle
reg_1_neurons = generate_uniform_neurons_on_circle(num_reg_1_neurons)


# Code will remove an output neuron if its average firing rate through time series is less than or equal to
# laziness_threshold * (average of average firing rates of output neurons)
laziness_threshold = 0.02


########################################################################
########################################################################

for fixed_number_output in list_num_reg_2_neurons

    mkpath("output_pop_size_$fixed_number_output")
    
    for trial_index in 1:num_trials

        try

            # Construct bias vector
            bias_vector = zeros(fixed_number_output)
            for row_index in axes(bias_vector)[1]
                bias_vector[row_index] = 0.0
            end

            num_reg_2_neurons = fixed_number_output
            
            for learning_step_index in 1:num_learning_steps

                println("Working on output size $fixed_number_output, trial $trial_index, learning step $learning_step_index")
                
                println(now())
                
                flush(stdout)
                
                # Generate random skipping circular walk
                path_on_circle = generate_random_skipping_circular_walk(num_walks, num_steps_per_walk, step_size_range)

                # Create region 1 neural response matrix
                reg_1_neu_resp_matrix = calculate_neural_response_matrix(reg_1_neurons, path_on_circle, response_function)

                # Add noise to region 1 neural response matrix
                reg_1_neu_resp_matrix = add_normal_random_noise(reg_1_neu_resp_matrix, 0.05, 0.025)

                # Create and save input dictionary
                input_dictionary = Dict(
                    "num_reg_1_neurons" => num_reg_1_neurons,
                    "max_rate" => max_rate,
                    "slope" => slope,
                    "reg_1_neurons" => reg_1_neurons,
                    "path_on_circle" => path_on_circle,
                    "reg_1_neu_resp_matrix" => reg_1_neu_resp_matrix
                )
                save("output_pop_size_$fixed_number_output/trial_$trial_index/learning/step_$learning_step_index/input_dictionary.jld2", "dictionary", input_dictionary)
                
                # Save TRANSPOSE of region 1 neural response matrix as h5
                # Need transpose because of how Python will read it to compute dissimilarities
                h5write("output_pop_size_$fixed_number_output/trial_$trial_index/learning/step_$learning_step_index/reg_1_response_matrix.h5", "raster", copy(transpose(reg_1_neu_resp_matrix)))

                # Run similarity multiprocessing on region 1
                reg_1_raster_path = "output_pop_size_$fixed_number_output/trial_$trial_index/learning/step_$learning_step_index/reg_1_response_matrix.h5"
                reg_1_distance_path = "output_pop_size_$fixed_number_output/trial_$trial_index/learning/step_$learning_step_index/reg_1_distance.h5"
                run(`python3 compute_similarity_multiprocessing.py $reg_1_raster_path 100 $reg_1_distance_path`)


                # Load connection matrix
                connection_matrix = h5read("output_pop_size_$fixed_number_output/trial_$trial_index/learning/step_$learning_step_index/step_$learning_step_index"*"_connection_matrix.h5", "matrix")


                # Calculate region 2 neural reponse matrix
                reg_2_neu_resp_matrix = calculate_output_neural_response_matrix(reg_1_neu_resp_matrix, connection_matrix, bias_vector)

                # Add noise to region 2 neural response matrix
                reg_2_neu_resp_matrix = add_normal_random_noise(reg_2_neu_resp_matrix, 0.05, 0.025)

                # REMOVE LAZY NEURONS FROM REGION 2 NEURAL RESPONSE MATRIX
                lazy_neuron_row_indices = []
                # Find the average firing rate of each neuron through time series
                row_averages = mean(reg_2_neu_resp_matrix, dims = 2)
                avg_reg_2_firing_rate = mean(row_averages)
                
                # Remove neurons
                for row_index in 1:num_reg_2_neurons
                    row_average = row_averages[row_index]
                    if row_average <= avg_reg_2_firing_rate*laziness_threshold
                        append!(lazy_neuron_row_indices, row_index)
                    end
                end
                reg_2_neu_resp_matrix = reg_2_neu_resp_matrix[1:end .∉[lazy_neuron_row_indices], 1:end]

                # Update number of region 2 neurons
                num_reg_2_neurons = size(reg_2_neu_resp_matrix)[1]

                # Remove unwanted rows from connection matrix
                connection_matrix = connection_matrix[1:end .∉[lazy_neuron_row_indices], 1:end]

                # Remove unwated rows from bias vector
                bias_vector = bias_vector[1:end .∉[lazy_neuron_row_indices], 1:end]


                # Save TRANSPOSE of region 1 neural response matrix as h5
                # Need transpose because of how Python will read it to compute dissimilarities
                h5write("output_pop_size_$fixed_number_output/trial_$trial_index/learning/step_$learning_step_index/reg_2_response_matrix.h5", "raster", copy(transpose(reg_2_neu_resp_matrix)))

                # Run similarity multiprocessing on region 2
                reg_2_raster_path = "output_pop_size_$fixed_number_output/trial_$trial_index/learning/step_$learning_step_index/reg_2_response_matrix.h5"
                reg_2_distance_path = "output_pop_size_$fixed_number_output/trial_$trial_index/learning/step_$learning_step_index/reg_2_distance.h5"
                run(`python3 compute_similarity_multiprocessing.py $reg_2_raster_path 100 $reg_2_distance_path`)

                # Pause for 1 second because last computation can be too fast on Pegasus
                sleep(1)

                # Run cross-similarity
                conj_rate_distance_path = "output_pop_size_$fixed_number_output/trial_$trial_index/learning/step_$learning_step_index/conj_rate_distance.h5"
                run(`python3 cross_similarity_multiprocessing.py $reg_1_raster_path $reg_2_raster_path 100 $conj_rate_distance_path`)


                # Load cross-dissimilarity matrix where rows are indexed by region 2 neurons
                cross_dissim_matrix = h5read("output_pop_size_$fixed_number_output/trial_$trial_index/learning/step_$learning_step_index/conj_rate_distance.h5", "distance")


                # Get vector of averages of each row of cross_dissim_matrix
                vector_of_mean_distances = mean(cross_dissim_matrix, dims = 2)


                # Copy old connection matrix
                updated_connection_matrix = copy(connection_matrix)


                # Update connection matrix
                for row_index in 1:num_reg_2_neurons
                    mean = vector_of_mean_distances[row_index]
                    for col_index in 1:num_reg_1_neurons
                        dissimilarity = cross_dissim_matrix[row_index, col_index]
                        if dissimilarity <= (1-learning_threshold)*mean || dissimilarity >= (1+learning_threshold)*mean
                            difference = mean - dissimilarity
                            current_connection_weight = connection_matrix[row_index, col_index]
                            if current_connection_weight != 0
                                updated_connection_weight = current_connection_weight + learning_step_size*difference
                                if updated_connection_weight <= 0
                                    updated_connection_matrix[row_index, col_index] = max(-1, updated_connection_weight)
                                else 
                                    updated_connection_matrix[row_index, col_index] = min(updated_connection_weight, 1)
                                end
                            end
                        end
                    end
                end


                # Save updated connection matrix
                next_learning_step_index = learning_step_index + 1
                mkpath("output_pop_size_$fixed_number_output/trial_$trial_index/learning/step_$next_learning_step_index")
                h5write("output_pop_size_$fixed_number_output/trial_$trial_index/learning/step_$next_learning_step_index/step_$next_learning_step_index"*"_connection_matrix.h5", "matrix", updated_connection_matrix)
                heatmap(updated_connection_matrix)
                png("output_pop_size_$fixed_number_output/trial_$trial_index/learning/step_$next_learning_step_index/step_$next_learning_step_index"*"_connection_matrix.h5")
                # heatmap(connection_matrix)
                # png("output_pop_size_$fixed_number_output/trial_$trial_index/learning/step_$next_learning_step_index/step_$next_learning_step_index"*"_connection_matrix")
            end

        catch
            println("Error on output size $fixed_number_output, trial $trial_index")
            flush(stdout)
            continue
        end
    end
end