#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <time.h>
#include <numeric>
#include <cmath>
#include "bmp_utility.h"



// OpenCL header file and Intel FPGA SDK header file
// Uncomment when running FPGA Kernel
//#include "CL/opencl.h"
//#include "AOCLUtils/aocl_utils.h" 

// Toggle this to run on CPU (DE1-SoC or your machine) and FPGA
// 0 to run on CPU
// 1 to run on FPGA
#define FPGA 0




#if FPGA == 1// If FPGA == 1, OpenCL related code will execute

    // namespace for Intel FPGS SDK
    using namespace aocl_utils;

    // OpenCL runtime configuration
    cl_platform_id platform = NULL;


    unsigned num_devices = 0;
    scoped_array<cl_device_id> device;
    cl_context context = NULL;
    scoped_array<cl_command_queue> queue;
    cl_program program = NULL;
    scoped_array<cl_kernel> kernel;

    scoped_array<unsigned> n_per_device; // num_devices elements
    cl_program Program;


#endif






// Image size in 1D array = 28 x 28
const int inputSize = 784; // 28x28 input image


// Number of neurons in the hidden layer
// This code assumes a single hidden layer
// If you use more layers modify accordingly
const int numNeurons = 10;

// Tile size to perform matrix multiplication
// Experiment with this size and report those values 
const int inputTileSize = 28;
    


// Variables to hold input data
std::vector<float> image_data; // image data



std::vector<float> hidden_layer1_weights;
std::vector<float> hidden_layer1_biases;
std::vector<float> hidden_layer1_out;



std::vector<float> output_layer_weights;
std::vector<float> output_layer_biases;
std::vector<float> output_layer_out;



#if FPGA == 1

// Neural network buffers
// We transfer data to corresponding buffers before launching the Kernel


scoped_array<cl_mem> input_image_buf(num_devices); // Buffer for the input image
scoped_array<cl_mem> hidden_layer1_weights_buf(num_devices); // Buffers for layer1 weigths




scoped_array<cl_mem> output_layer_weights_buf(num_devices); // Buffers for layer2 weights
scoped_array<cl_mem> layer_output_buf(num_devices); // Buffers for the output of each layer

#endif


std::string layer1_weightsPath = "fc1_weight.bin";
std::string layer1_biasesPath = "fc1_bias.bin";

std::string output_weightsPath = "fc2_weight.bin";
std::string output_biasesPath = "fc2_bias.bin";

#if FPGA == 1 // functions that setup opencl environment, problem, run and cleanup 
            // needed only when running the kernel   
bool init_opencl();
void run();
void cleanup();
#endif

void normalizeImage(unsigned char* imageData, size_t imageSize, std::vector<float>& normalizedImage);
void setupDataAndModels();
void run_cpu();
void processTiles_weightStatinary_CPU(int numNeurons,
    int inputSize, // Size of the input array
    int inputTileSize,  // Tile size of the Input vector          
    std::vector<float>& weights, // Weights array
    std::vector<float>& biases,  // biases array
    std::vector<float>& inputs,  // inputs array 
    std::vector<float>& outputs  // outputs array);
    );
void cleanup_cpu();

void relu(std::vector<float>& v);

int getMaxIn(std::vector<float>& v);


bool loadModelParameters(const std::string& weightsPath, const std::string& biasesPath, 
                         std::vector<float>& weightsBuffer, std::vector<float>& biases);


std::vector<float> loadFloatsFromFile(const std::string& filename);
void log_softmax(std::vector<float>& v);


// Code execution starts here
int main(int argc, char **argv) {

  #if FPGA == 1
  // Options from base OpenCL code, ignore for this lab  
  Options options(argc, argv);

  // Optional argument to specify the problem size.
  if(options.has("n")) {
    N = options.get<unsigned>("n");
  }

  // Initialize OpenCL.
  if(!init_opencl()) {
    return -1;
  }
  #endif

  setupDataAndModels();


  // Run the kernel.
  #if FPGA == 1
  //Initialize the problem data.
  run();
  #else
  run_cpu();
  #endif

  // Free the resources allocated
  #if FPGA == 1
    cleanup();
  #else
    cleanup_cpu();
  #endif


  return 0;
}



void setupDataAndModels(){
    const char* filename = "first_image_mnist.bmp";
    int width = 0;
    int height = 0;

    unsigned char* pre_image_data = loadBMPGrayscale(filename, &width, &height);
    flipImageVertically(pre_image_data, width, height);

    normalizeImage(pre_image_data, width*height, image_data);
    
    printf("done loading image:%d\n",width*height);


    if (!loadModelParameters(layer1_weightsPath,layer1_biasesPath,hidden_layer1_weights,hidden_layer1_biases)) {

        std::cerr << "Failed to load model layer 1 parameters." << std::endl;
        #if FPGA == 1
            cleanup();
        #endif
        return;
    }

    if (!loadModelParameters(output_weightsPath,output_biasesPath,output_layer_weights,output_layer_biases)) {
        std::cerr << "Failed to load model output layer parameters." << std::endl;
        #if FPGA == 1
            cleanup();
        #endif
        return;
    }

    printf("loaded model parameters\n");



}


void log_softmax(std::vector<float>& v) {

    float maxElement = *std::max_element(v.begin(), v.end());
    std::vector<float> exp_values(v.size());
    float sum = 0.0f;

    // Calculate exponentials and sum them
    for(size_t i = 0; i < v.size(); ++i) {
        exp_values[i] = std::exp(v[i] - maxElement);
        sum += exp_values[i];
    }

    // Normalize
    for(size_t i = 0; i < v.size(); ++i) {
        v[i] = std::log(exp_values[i] / sum);
    }



}


void normalizeImage(unsigned char* imageData, size_t imageSize, std::vector<float>& normalizedImage) {
    normalizedImage.resize(imageSize);

    float mean=0.1307f;
    float std=0.3081f;
    for (size_t i = 0; i < imageSize; ++i) {
        normalizedImage[i] = (imageData[i] / 255.0f - mean) / std;
    }
}


std::vector<float> loadFloatsFromFile(const std::string& filename) {
    // Open the file in binary mode
    std::ifstream file(filename.c_str(), std::ios::binary | std::ios::ate);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {}; // Return an empty vector in case of failure
    }

    // Determine the file size
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Calculate the number of float elements
    std::streamsize numElements = size / sizeof(float);

    std::vector<float> buffer(numElements);

    // Read the file content into the buffer
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cerr << "Failed to read floats from file: " << filename << std::endl;
        return {}; // Return an empty vector in case of failure
    }

    file.close(); // Close the file
    return buffer; // Return the loaded floats
}





bool loadModelParameters(const std::string& weightsPath, const std::string& biasesPath, 
                         std::vector<float>& weightsBuffer, std::vector<float>& biases) {


    weightsBuffer = loadFloatsFromFile(weightsPath);


    biases = loadFloatsFromFile(biasesPath);


    return true; // Successfully loaded and transferred weights and biases
}


#if FPGA == 1
bool init_opencl() {
  cl_int status;

  printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL device.
  device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  printf("Platform: %s\n", getPlatformName(platform).c_str());
  printf("Using %d device(s)\n", num_devices);
  for(unsigned i = 0; i < num_devices; ++i) {
    printf("  %s\n", getDeviceName(device[i]).c_str());
  }


// Now you can use `device` in clGetDeviceInfo
cl_ulong max_mem_alloc_size;
cl_int err = clGetDeviceInfo(device[0], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);

if (err != CL_SUCCESS) {
    // Error handling
    std::cerr << "Error getting device info: " << err << std::endl;
} else {
    std::cout << "Max memory allocation size: " << max_mem_alloc_size << " bytes" << std::endl;
}


  // Create the context.
  context = clCreateContext(NULL, num_devices, device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  // Load and build programs for each kernel
  std::string hiddenLayerBinary = getBoardBinaryFile("hidden_emulation", device[0]);
  //std::string outputLayerBinary = getBoardBinaryFile("output_layer", device[0]);
  printf("Using AOCX: %s\n", hiddenLayerBinary.c_str());
  //printf("Using AOCX: %s\n", outputLayerBinary.c_str());
  printf("creating program\n");
  // Create program for hidden layer
  //std::cout << device.c_str() << std::endl;
  printf("num devices:%d\n",num_devices);
  hiddenLayerProgram = createProgramFromBinary(context, hiddenLayerBinary.c_str(), device, num_devices);
  printf("done creating program\n");
  status = clBuildProgram(hiddenLayerProgram, 0, NULL, NULL, NULL, NULL);
  checkError(status, "Failed to build hidden layer program");


  // Create kernel for hidden layer
  cl_kernel hiddenLayerKernel = clCreateKernel(hiddenLayerProgram, "hidden", &status);
  checkError(status, "Failed to create hidden layer kernel");

  printf("loaded kernel\n");


  // Create per-device objects.
  queue.reset(num_devices);
  kernel.reset(num_devices); // Two kernels per device for simplicity
  n_per_device.reset(num_devices);


  // Sizes of neural network
    const size_t input_layer_size = 784; // MNIST 28x28 images
    const size_t hidden_layer_size = 10; // a single hidden layer with 32 neurons
    const size_t output_layer_size = 10; // for MNIST digit classification (0-9)

    // Calculate the total size for weights and biases
    const size_t weights_hidden_size = input_layer_size * hidden_layer_size;
    const size_t biases_hidden_size = hidden_layer_size;
    const size_t weights_output_size = hidden_layer_size * output_layer_size;
    const size_t biases_output_size = output_layer_size;

  


  for(unsigned i = 0; i < num_devices; ++i) {
    // Command queue.
    queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    // Kernel.
    kernel[i] = hiddenLayerKernel;
    //kernel[i * 2 + 1] = outputLayerKernel;

    // Determine the number of elements processed by this device.
    n_per_device[i] = N / num_devices; // number of elements handled by this device

    // Spread out the remainder of the elements over the first
    // N % num_devices.
    if(i < (N % num_devices)) {
      n_per_device[i]++;
    }

    printf("completed setting up command queue\n");

#if USE_SVM_API == 0
    // Input buffers.

    printf("started creating input image buffers\n");

#else
    cl_device_svm_capabilities caps = 0;

    status = clGetDeviceInfo(
      device[i],
      CL_DEVICE_SVM_CAPABILITIES,
      sizeof(cl_device_svm_capabilities),
      &caps,
      0
    );
    checkError(status, "Failed to get device info");

    if (!(caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) {
      printf("The host was compiled with USE_SVM_API, however the device currently being targeted does not support SVM.\n");
      // Free the resources allocated
      cleanup();
      return false;
    }
#endif /* USE_SVM_API == 0 */
  }

  printf("done creating buffers\n");

  return true;

}



#if FPGA == 1
void processTiles_weightStatinary(int numNeurons,
    int inputSize, // Size of the input array
    int inputTileSize,  // Tile size of the Input vector          
    std::vector<float>& weights, // Weights array
    std::vector<float>& biases,  // biases array
    std::vector<float>& inputs,  // inputs array 
    std::vector<float>& outputs  // outputs array
    ) {

    printf("in the weight stationary function\n");

    cl_int err;

    int outputNeuronsTileSize = 10;
    int currentTileSize = inputTileSize;

    #if FPGA == 1
        cl_mem weightsTileBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, weightsPerTile * sizeof(signed char), NULL, &err);
        
        //#TODO : create remaining required buffers

        if(err != CL_SUCCESS){
        }else{
            printf("done creating buffer\n");
        }
    #else
    #endif

    #if FPGA == 1    
        clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&inputTileBuffer);
        //#TODO : set remaining kernel arguments
    #endif


    //#TODO: similar to weightstationary_cpu code implemnt the same logic with inner loop taken care by the parallel kernes

    //For each kernel launch you write data to the buffers using a command similar to the following 
    // err = clEnqueueWriteBuffer(queue, weightsTileBuffer, CL_TRUE, 0, weightsPerTile * sizeof(signed char), &hidden_layer1_weights[weightsStartIndex], 0, NULL, NULL);

    //#TODO: After writing buffers to each kernel launch kernel using the follwoing code

    // global work size and local work sizes are the kernel dimensions, for this lab sizes as simple as the following is good

    //size_t global_work_size[] = {static_cast<size_t>(10)};
    //size_t local_work_size[] = {static_cast<size_t>(1)};

    // assuming you implemented a 1D kernel in OpenCL, if you implemented 2D please discuss with TA    

    //err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, local_work_size, 1, NULL, NULL);

    // OpenCL kernels running on FPGA are not synchornous. You will synchronize your computations by waiting till the queue is finished by the following code    
    // clFinish(queue);    
    

    #if FPGA == 1
        clReleaseMemObject(inputTileBuffer);
        //#TODO: release remaining memory buffers
    #endif
}
#endif




void run() {
    cl_int status;

    // hidden 
    const unsigned hiddenLayerIndex = 0;
    const unsigned outputLayerIndex = 1;

    size_t hidden_weights_offset = 0; // Starting at the beginning for the hidden layer
    size_t output_weights_offset = 784 * 10; // Adjust based on your buffer organization
    size_t hidden_biases_offset = 0;
    size_t output_biases_offset = 10; // Assuming biases are also concatenated

    size_t hidden_work_size = {10};
    size_t output_work_size = {10};


    printf("started running on fpga\n");

    //#TODO: similar to connecting computing each layer and connecting them in CPU code, implement same logic here but calling the FPGA functions
}
#endif

void matrixMulCPU(
    std::vector<float>& input_tile,  // Tile of the Input vector
    std::vector<float>& weights_tile, // Tile of the Weights matrix
    int input_tile_size,                  // Size of the input tile
    int output_neurons_tile_size,         // Size of the output tile (number of neurons in this tile)
    std::vector<float>& output_tile                // Output vector tile
){

    int neuron_id = 0;
    for(neuron_id;neuron_id<output_neurons_tile_size;neuron_id++){
    // Ensure we don't process more neurons than we have in this tile
    if (neuron_id < output_neurons_tile_size) {
        float temp_sum = 0.0;
        
        // Compute the dot product of the input tile and the corresponding weights
        for (int i = 0; i < input_tile_size; ++i) {

          int weight_index = neuron_id * input_tile_size + i;
          //printf("weight index:%d\n",weight_index);
          temp_sum += (float)input_tile[i] * (float)weights_tile[weight_index];
     }

        // Write the computed sum for this neuron to the output tile
        output_tile[neuron_id] += temp_sum;
    }
    }

}


std::vector<float> loadWeights(int weightsStartIndex,int numNeurons,int inputTileSize,int inputSize,
    std::vector<float>& weights,std::vector<float>& temp_wts){
    
    int index = 0;
    for(int i=0;i<numNeurons;i++){
        for(int j=0;j<inputTileSize;j++){
            temp_wts[index] = weights[(i)*inputSize + j+weightsStartIndex];
            //printf("index:%d\n",index);
            index++;
        }
    }
    return temp_wts;    
}


void processTiles_weightStatinary_CPU(
    int numNeurons,
    int inputSize, // Size of the input array
    int inputTileSize,  // Tile size of the Input vector          
    std::vector<float>& weights, // Weights array
    std::vector<float>& biases,  // biases array
    std::vector<float>& inputs,  // inputs array 
    std::vector<float>& outputs  // outputs array
    ) {

    printf("in the weight stationary function of CPU\n");    

    int numTiles = inputSize / inputTileSize; // Ensure this division is an integer
    int totalWeights = inputSize * numNeurons;
    int weightsPerTile = numNeurons*inputTileSize; // Assuming an even distribution of neurons per tile



    for (int tileIndex = 0; tileIndex < numTiles; ++tileIndex) {
        
        int weightsStartIndex = tileIndex * inputTileSize; 
        std::vector<float> temp_wts;
        temp_wts.resize(numNeurons * inputTileSize);
        loadWeights(weightsStartIndex,numNeurons,inputTileSize,inputSize,weights,temp_wts);

        std::vector<float> inputSlice(std::next(inputs.begin(), weightsStartIndex), std::next(inputs.begin(), weightsStartIndex+inputTileSize));

        matrixMulCPU(
            inputSlice,  // Tile of the Input vector
            temp_wts, // Tile of the Weights matrix
            inputTileSize,                  // Size of the input tile
            numNeurons,         // Size of the output tile (number of neurons in this tile)
            outputs
        );
    }

    for(int i=0;i<numNeurons;i++){
        outputs[i] += biases[i];
    } 

}

void run_cpu() {
    

    // hidden 
    const unsigned hiddenLayerIndex = 0;
    const unsigned outputLayerIndex = 1;

    size_t hidden_weights_offset = 0; // Starting at the beginning for the hidden layer
    size_t output_weights_offset = 784 * 10; // Adjust based on your buffer organization
    size_t hidden_biases_offset = 0;
    size_t output_biases_offset = 10; // Assuming biases are also concatenated

    size_t hidden_work_size = {10};
    size_t output_work_size = {10};


    printf("started running on CPU\n");

    hidden_layer1_out.resize(numNeurons * inputTileSize);
    processTiles_weightStatinary_CPU(hidden_work_size,
    inputSize, // Size of the input array
    inputTileSize,  // Tile size of the Input vector          
    hidden_layer1_weights, // Weights array
    hidden_layer1_biases,  // biases array
    image_data,  // inputs array 
    hidden_layer1_out  // outputs array);
    );

    //printf("done first layer: %d\n",hidden_layer1_out.size());

    std::cout << "Output of fc1 (before ReLU): ";

    for(int i=0;i<10;i++){
        std::cout << hidden_layer1_out[i] << " ";
    }
    std::cout << std::endl;

    relu(hidden_layer1_out);

    std::cout << "Output of fc1 (after ReLU): ";

    for(int i=0;i<10;i++){
        std::cout << hidden_layer1_out[i] << " ";
    }
    std::cout << std::endl;

    output_layer_out.resize(numNeurons * inputTileSize);

    processTiles_weightStatinary_CPU(output_work_size,
    output_work_size, // Size of the input array
    10,  // Tile size of the Input vector          
    output_layer_weights, // Weights array
    output_layer_biases,  // biases array
    hidden_layer1_out,  // inputs array 
    output_layer_out  // outputs array);
    );


    std::cout << "Output of fc2 (before LogSoftmax): "; 
    for(int i=0;i<10;i++){
        std::cout << output_layer_out[i] << " ";
    }
    std::cout << std::endl;

    log_softmax(output_layer_out);


    std::cout << "Output of fc2 (after LogSoftmax): "; 
    for(int i=0;i<10;i++){
        std::cout << output_layer_out[i] << " ";
    }
    std::cout << std::endl;

    int Label = getMaxIn(output_layer_out);
    printf("Predicted label:%d\n",Label);

}

int getMaxIn(std::vector<float>& v){
    int maxIndex = std::distance(v.begin(), std::max_element(v.begin(), v.end()));
    return maxIndex;
}

void relu(std::vector<float>& v) {
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = std::max(0.0f, v[i]);
    }
}

void cleanup_cpu() {

}


void cleanup() {
    #if FPGA == 1
    cl_int status;

    // Release kernels
    for(unsigned i = 0; i < num_devices; ++i) { // Assuming 2 kernels per device
        if(kernel && kernel[i]) {
            status = clReleaseKernel(kernel[i]);
            checkError(status, "Failed to release kernel");
        }
    }

    Release programs
    Assuming you have stored your program objects similar to kernels
    For example, if you have two program objects like hiddenLayerProgram and outputLayerProgram
    if(hiddenLayerProgram) {
        status = clReleaseProgram(hiddenLayerProgram);
        checkError(status, "Failed to release hidden layer program");
    }
    if(outputLayerProgram) {
        status = clReleaseProgram(outputLayerProgram);
        checkError(status, "Failed to release output layer program");
    }

    // Finally, release the context
    if(context) {
        status = clReleaseContext(context);
        checkError(status, "Failed to release context");
    }
    #endif

    // If you have other resources allocated, make sure to release them properly
}




