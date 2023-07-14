using .barcode_script
using PyCall

circ_coords = pyimport("Circ_coords_script.py")


iterations = 0      ##### number of times you want to iterate a specific trial
input_neurons = 0   ##### number of input neurons
output_neurons = 0  ##### number of output neurons
max_output_neurons = 0  ##### for a range of output neurons, max number of neurons
step_size = 0   ##### desired increase of output neurons per trial

function main(iterations=1, input_neurons, output_neurons, max_output_neurons=0, step_size=0)
    

end