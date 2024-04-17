__kernel void matrixMul(
   __global const float* input_tile,  // Tile of the Input vector
   __global const float* weights_tile, // Tile of the Weights matrix
   const int input_tile_size,                  // Size of the input tile
   const int output_neurons_tile_size,         // Size of the output tile (number of neurons in this tile)
   __global float* output_tile                // Output vector tile
    )
{
    // Calculate the ID of the neuron this thread will compute
    int neuron_id = get_global_id(0);

    // Ensure we don't process more neurons than we have in this tile
    if (neuron_id < output_neurons_tile_size) {
        // Specify number of parallel units needed (by loop unrolling)
        //#pragma  unroll;
        float output = 0;
        for(int i = 0; i<input_tile_size; i++){
            output += input_tile[i] * weights_tile[input_tile_size*neuron_id+ i];
        }
        output_tile[neuron_id] = output;
    }
}