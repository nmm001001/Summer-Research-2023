include("barcode_script.jl")
using .barcode_script
# ENV["PYTHON"] = "venv/Scripts/python.exe"
# using Pkg
# Pkg.build("PyCall")
# using PyCall

# circ_coords = pyimport("Circ_coords_script")


iterations = 0      # number of times you want to iterate a specific trial
input_neurons = 0   # number of input neurons
output_neurons = 0  # number of output neurons
max_output_neurons = 0  # for a range of output neurons, max number of neurons
step_size = 0   # desired increase of output neurons per trial

function singleLayerMain(iterations, input_neurons, output_neurons, max_output_neurons, step_size)
    for output_step in output_neurons:step_size:max_output_neurons

        for iteration in 1:iterations
            DP, VRP, reg_1_neu_resp_matrix = generate_DP(input_neurons)
               
            conn_matrix = generate_FAT_ID_matrix(output_step)

            DQ, VRQ, reg_2_neu_response_matrix = generate_DQ(conn_matrix, reg_1_neu_resp_matrix)

            DQP, DPQ = compute_DQP_DPQ()

            VRP, VRQ, WP, WQ = VR_and_witness_comp(DP, DQ, DPQ, DQP)

            barcode_VRP, barcode_VRQ, barcode_WP, barcode_WQ = barcode_VR_Wit_comp(VRP, VRQ, WP, WQ)

            witness_bar = compute_largest_bar(barcode_WP)

            analogous_bars(VRP, DP, VRQ, DQ, WP, witness_bar)

            run(`python main_python.py`)

        
        end
    end

end



DP, VRP, reg_1_neu_resp_matrix = generate_DP(50) #input neurons
               
conn_matrix1 = generate_antipodal_conn(50, 10, 25) #hidden layers
conn_matrix2 = generate_FAT_ID_matrix(60,100)
conn_matrix3 = generate_FAT_ID_matrix(50,60)


DQ1, VRQ1, reg_2_neu_response_matrix1 = generate_DQ(conn_matrix1, reg_1_neu_resp_matrix) # passing the information forward
DQ2, VRQ2, reg_2_neu_response_matrix2 = generate_DQ(conn_matrix2, reg_2_neu_response_matrix1)
DQ3, VRQ3, reg_2_neu_response_matrix3 = generate_DQ(conn_matrix3, reg_2_neu_response_matrix2)

DQP, DPQ = compute_DQP_DPQ()

VRP, VRQ, WP, WQ = VR_and_witness_comp(DP, DQ1, DPQ, DQP)

barcode_VRP, barcode_VRQ, barcode_WP, barcode_WQ = barcode_VR_Wit_comp(VRP, VRQ, WP, WQ)

witness_bar = compute_largest_bar(barcode_WP)

analogous_bars(VRP, DP, VRQ, DQ1, WP, witness_bar)

run(`python main_python.py`)

