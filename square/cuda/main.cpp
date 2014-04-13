//
//  main.cpp
//  cuda
//
//  Created by Gerald Stanje on 4/12/14.
//  Copyright (c) 2014 Gerald. All rights reserved.
//

#include <iostream>
#include <vector>
#include <CUDA/CUDA.h>
using namespace std;

/*#define GPU_CALC

void cpu_square(vector<int> &in, vector<int> &out) {
    for(int i = 0; i < in.size(); i++) {
        out[i] = in[i] * in[i];
    }
}
*/

// Kernel:
__global__ void square(float* numbers) {
    // get the array coordinate:
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    // square the number:
    numbers[x] = numbers[x] * numbers[x];
}

int main(int argc, const char * argv[])
{
    // insert code here...
    /*vector<int> in;
    vector<int> out;
    
    for(int i = 0; i < 64; i++) {
        in.push_back(rand() % 100);
    }
    
    out.resize(in.size());
    
    cpu_square(in, out);
    */
    
    const unsigned int N = 100; // N numbers in array
    
    float data[N]; // array that contains numbers to be squared
    float squared[N]; // array to be filled with squared numbers
    
    // number to be squared will be the index:
    for(unsigned i=0; i<N; i++)
        data[i] = static_cast<float>(i);
    
    // allocate memory on CUDA device:
    float* pDevData; // pointer to the data on the CUDA Device
    cudaMalloc((void**)&pDevData, sizeof(data));
    
    // copy data to CUDA device:
    cudaMemcpy(pDevData, &data, sizeof(data), cudaMemcpyHostToDevice);
    
    // execute kernel function on GPU:
    square<<<10, 10>>>(pDevData);
    
    // copy data back from CUDA Device to ’squared’ array:
    cudaMemcpy(&squared, pDevData, sizeof(squared), cudaMemcpyDeviceToHost);
    
    // free memory on the CUDA Device:
    cudaFree(pDevData);
    
    // output results:
    for(unsigned i=0; i<N; i++)
        std::cout<<data[i]<<"^2 = "<<squared[i]<<"\n";
    
    return 0;
}

