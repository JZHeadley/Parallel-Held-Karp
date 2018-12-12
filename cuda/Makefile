
all: cuda-tsp

cuda-tsp:
	nvcc -std=c++11 -O3 -lpthread cuda-tsp.cu --cudart static --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61 -o $@

clean: 
	rm -Rf cuda-tsp
