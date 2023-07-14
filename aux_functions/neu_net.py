import numpy as np
import random


from sklearn.preprocessing import normalize



from aux_functions import utilities



# Generates inputs and outputs with connection matrix constructed by *adding* connections at a fixed angle theta = 2pi/'theta_denom' to identity matrix


def generate_inputs_outputs(linear_threshold, response_fn_slope, percent_removed):
    # Basic parameters
    num_reg_1_neurons = 100
    num_reg_2_neurons = 100
    num_steps = 12000
    winding_num = 7
    response_function_type = "linear relu"
    



    # Possible response function parameters
    # heavyside_parameter = 0.03
    max_rate = 1
    slope = response_fn_slope


    # Declar response function
    response_function = utilities.linear_relu_response_function(max_rate, slope)


    # Generate region 1 neurons, uniformly chosen on circle
    reg_1_neurons = utilities.generate_uniform_neurons_on_circle(num_reg_1_neurons)



    # Generate an evenly distributed circular path
    path_on_circle = utilities.generate_circular_path(winding_num, num_steps)

    # Create region 1 neural response matrix
    reg_1_neu_resp_matrix = utilities.calculate_neural_response_matrix(reg_1_neurons, path_on_circle, response_function)

    # Add noise to region 1 neural response matrix
    for row_index in range(num_reg_1_neurons):
        for column_index in range(num_steps):
            entry = reg_1_neu_resp_matrix[row_index, column_index]
            if entry != 0.0:
                entry_with_noise = np.random.normal(entry, 0.05*entry)
                entry_with_noise = max(0, entry_with_noise)
                reg_1_neu_resp_matrix[row_index, column_index] = entry_with_noise
            else:
                entry_with_noise = np.random.normal(0, 0.025)
                entry_with_noise = max(0, entry_with_noise)
                reg_1_neu_resp_matrix[row_index, column_index] = entry_with_noise


    # Create connection matrix
    connection_matrix = np.identity(num_reg_1_neurons)
    num_connections_to_remove = int(np.floor((percent_removed/100)*num_reg_1_neurons))
    cols_to_remove = random.sample(range(num_reg_1_neurons), num_connections_to_remove)
    for col_index in cols_to_remove:
        connection_matrix[col_index, col_index] = 0
    # connection_matrix = normalize(connection_matrix, axis = 1, norm = 'l1')


    # Create region 2 neural response matrix
    reg_2_neu_resp_matrix = np.matmul(connection_matrix, reg_1_neu_resp_matrix)
    
    # Apply linear threshold to region 2 neural response matrix
    reg_2_neu_resp_matrix = utilities.apply_threshold_linear(reg_2_neu_resp_matrix, linear_threshold)

    # Add noise to region 2 neural response matrix
    for row_index in range(num_reg_2_neurons):
        for column_index in range(num_steps):
            entry = reg_2_neu_resp_matrix[row_index, column_index]
            if entry != 0.0:
                entry_with_noise = np.random.normal(entry, 0.05*entry)
                entry_with_noise = max(0, entry_with_noise)
                reg_2_neu_resp_matrix[row_index, column_index] = entry_with_noise
            else:
                entry_with_noise = np.random.normal(0, 0.025)
                entry_with_noise = max(0, entry_with_noise)
                reg_2_neu_resp_matrix[row_index, column_index] = entry_with_noise
    
    

    # Creat output dictionary
    output_dictionary = {
        'num_reg_1_neurons': num_reg_1_neurons,
        'num_reg_2_neurons': num_reg_2_neurons,
        'num_steps': num_steps,
        'winding_num': winding_num,
        'max_rate': max_rate,
        'slope': slope,
        'response_function type': response_function_type,
        'reg_1_neurons': reg_1_neurons,
        'path_on_circle': path_on_circle,
        'reg_1_neu_resp_matrix': reg_1_neu_resp_matrix,
        'connection_matrix': connection_matrix,
        'reg_2_neu_resp_matrix': reg_2_neu_resp_matrix
    }


    return output_dictionary, reg_1_neu_resp_matrix, reg_2_neu_resp_matrix


