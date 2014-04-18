#include <iostream>
#include <iomanip>
#include <vector>
#include <CUDA/CUDA.h>
using namespace std;

// Kernel:
__global__ void square(float *d_out, float *d_in) {
    int idx = threadIdx.x;
	float f = d_in[idx];
    
    d_out[idx] = f * f;
}

int main(int argc, const char * argv[])
{	
    const unsigned int ARRAY_SIZE = 64; // N numbers in array
    const unsigned int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
	
    float h_in[ARRAY_SIZE]; // array that contains numbers to be squared
    float h_out[ARRAY_SIZE]; // array to be filled with squared numbers
    
    // number to be squared will be the index:
    for(unsigned i=0; i<ARRAY_SIZE; i++) {
        h_in[i] = static_cast<float>(i);
    }
	
    // allocate memory on CUDA device:
    float *d_in; // pointer to the data on the CUDA Device
	float *d_out; // pointer to the data on the CUDA Device
	
    cudaMalloc((void**)&d_in, ARRAY_BYTES);
    cudaMalloc((void**)&d_out, ARRAY_BYTES);
	
    // copy data to CUDA device:
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
    
    // execute kernel function on GPU:
    square<<<1, ARRAY_SIZE>>>(d_out, d_in);
    
    // copy data back from CUDA Device to ’squared’ array:
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    
    // output results:
    cout.setf(ios::fixed, ios::floatfield);
    cout.setf(ios::showpoint);  
    for(unsigned i=0; i<ARRAY_SIZE; i++) {
        std::cout << setprecision(2) << h_out[i];
		
    	if((i % 4) != 3) {
			cout << "\t";
		}
		else {
			cout << "\n";
		}
	}
	
    // free memory on the CUDA Device:
    cudaFree(d_in);
	cudaFree(d_out);
	
    cin.get();
    
    return 0;
}

