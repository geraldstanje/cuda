all: square
OBJS = square.cu
CC = nvcc
DEBUG = -g
CFLAGS = -O3 -c $(DEBUG) -stdlib=libstdc++
LDFLAGS = $(DEBUG) -lcudart
INCLUDES = -I/Developer/NVIDIA/CUDA-5.5/include/
CUDA_LIBS = -L/Developer/NVIDIA/CUDA-5.5/lib/

square : $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o square $(CUDA_LIBS)
	
square.o : square.cu
	$(CC) $(CFLAGS) square.cu $(INCLUDES)
	    
clean:
	rm -f *.o square; rm -r *.dSYM