#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
using namespace std;
using namespace thrust;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

typedef struct
{
	int id;
	double x;
	double y;
} City;

void printDistanceMatrix(float*h_distances, int numCities, int numFeatures);
double fRand(double fMin, double fMax);
vector<City> generateCities(int numCities, int gridDimX, int gridDimY);
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true);

__global__ void computeDistances(int numInstances, int numAttributes, float* dataset, float* distances)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int row = tid / numInstances; // instance1Index
	int column = tid - ((tid / numInstances) * numInstances); //instance2Index
	if ((tid < numInstances * numInstances))
	{
		float sum = 0;
		int instance1 = row * numAttributes;
		int instance2 = column * numAttributes;
		for (int atIdx = 1; atIdx < numAttributes; atIdx++) // start at 1 so we don't compare the id of each city
		{
			sum += ((dataset[instance1 + atIdx] - dataset[instance2 + atIdx]) * (dataset[instance1 + atIdx] - dataset[instance2 + atIdx]));
		}
		distances[row * numInstances + column] = (float) sqrt(sum);
		distances[column * numInstances + row] = distances[row * numInstances + column]; //set the distance for the other half of the pair we just computed
	}
}

__device__ unsigned int countNumBits(int n)
{
	unsigned int count = 0;
	while (n)
	{
		count += n & 1;
		n >>= 1;
	}
	return count;
}

__device__ unsigned long long curPosition;
__global__ void findPermutations(char* permutationsOfK, int k, unsigned long long lowerBound, unsigned long long upperBound)
{
	curPosition = 0;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int numToCheck = lowerBound + tid;
	unsigned int count = 0;
	unsigned int curBitPosition = 0;
	if (numToCheck < upperBound)
	{
		while (numToCheck)
		{
			if (numToCheck & 1)
			{
				atomicAdd(&curPosition, (float) 1);
				int permutationStartPos = curPosition * k;
				permutationsOfK[permutationStartPos + count] = curBitPosition;
				count++;
			}
			n >>= 1;
			curBitPosition++;
		}
//		if (countNumBits(numToCheck) == k)
//		{
//			atomicAdd(&curPosition, (float) 1);
//			int curIndex = curPosition * k;
//			for (int i = 0; i < k; i++)
//			{
//				permutationsOfK[curIndex + i] = tid;
//			}
//		}
	}
}

int main(void)
{
	float* d_distances;
	float* h_distances;
	float* h_dataset;
	float* d_dataset;
	char* d_permutations_of_k;

	int numCities = 6;
	int numFeatures = 3;
	int k = numCities % 2 == 0 ? numCities / 2 : (ceil(numCities / 2));

	cudaEvent_t allStart, allStop, distStart, distStop, permutationsStart, permutationsStop;
	cudaEventCreate(&allStart);
	cudaEventCreate(&allStop);
	cudaEventCreate(&distStart);
	cudaEventCreate(&distStop);
	cudaEventCreate(&permutationsStart);
	cudaEventCreate(&permutationsStop);

	float allMilliseconds = 0, distMilliseconds = 0, permutationMilliseconds = 0;

	vector<City> cities = generateCities(numCities, 500, 500);

	cudaMallocHost(&h_dataset, sizeof(float) * numCities * numFeatures);
	cudaMalloc(&d_dataset, sizeof(float) * numCities * numFeatures);

	cudaMallocHost(&h_distances, sizeof(float) * numCities * numCities);
	cudaMalloc(&d_distances, sizeof(float) * numCities * numCities);

	for (int i = 0; i < numCities; i++) // convert cities vector to the array the distance computation kernel expects
	{
		h_dataset[i * numFeatures] = cities[i].id; //cities[i].id;
		h_dataset[i * numFeatures + 1] = cities[i].x;
		h_dataset[i * numFeatures + 2] = cities[i].y;
	}

	cudaEventRecord(allStart);

	int threadsPerBlock = 1024;
	int blocksPerGrid = ((numCities * numCities) + threadsPerBlock - 1) / threadsPerBlock;

	cudaEventRecord(distStart);
	gpuErrchk(cudaMemcpy(d_dataset, h_dataset, numCities * numFeatures * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_distances, h_distances, numCities * numCities * sizeof(float), cudaMemcpyHostToDevice));
	computeDistances<<<blocksPerGrid, threadsPerBlock, 0>>>(numCities, numFeatures, d_dataset, d_distances);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaMemcpy(h_distances, d_distances, numCities * numCities * sizeof(float), cudaMemcpyDeviceToHost));
	cudaEventRecord(distStop);
	cudaEventSynchronize(distStop);
	threadsPerBlock = 256;
	int numPossibilities = pow(2, numCities) - pow(2, k - 1);
	blocksPerGrid = ((numPossibilities) + threadsPerBlock - 1) / threadsPerBlock;

	gpuErrchk(cudaFree(d_dataset));
	cudaEventRecord(permutationsStart);
	gpuErrchk(cudaMalloc(&d_permutations_of_k, pow(2, 29) * sizeof(char) * k));
	char *h_permutationsOfK = (char*) malloc(pow(2, 29) * sizeof(char) * k);

	for (int i = 0; i < numCities; i++)
	{
//		int i = 20;
		findPermutations<<<blocksPerGrid, threadsPerBlock, 0>>>(d_permutations_of_k, i, pow(2, i) - 1, pow(2, numCities));
//		unsigned long long finalPos;
//		gpuErrchk(cudaMemcpyFromSymbol(&finalPos, curPosition, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost));
//		finalPos++;
//		gpuErrchk(cudaMemcpy(h_permutationsOfK, d_permutations_of_k, pow(2, 29) * sizeof(char) * k, cudaMemcpyDeviceToHost));
//
	}
	gpuErrchk(cudaPeekAtLastError());
	cudaEventRecord(permutationsStop);
	cudaEventSynchronize(permutationsStop);
	cudaDeviceSynchronize();
	unsigned long long finalPos;
	gpuErrchk(cudaMemcpyFromSymbol(&finalPos, curPosition, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost));
	finalPos++;
	printf("found %llu permutations\n", (unsigned long long) finalPos);
	cudaEventRecord(allStop);
	cudaEventSynchronize(allStop);

	cudaEventElapsedTime(&distMilliseconds, distStart, distStop);
	cudaEventElapsedTime(&permutationMilliseconds, permutationsStart, permutationsStop);
	cudaEventElapsedTime(&allMilliseconds, allStart, allStop);

	printf("%i choose %i is %i\n", numCities, k, (int) finalPos);
	printf("The distance calculation for %i cities took %llu ms.\n", numCities, (long long unsigned int) distMilliseconds);
	printf("The permutations calculation for %i cities with %i size subsets took %llu ms.\n", numCities, k,
			(long long unsigned int) permutationMilliseconds);
	printf("The salesman traversed %i cities in %llu ms.\n", numCities, (long long unsigned int) allMilliseconds);

	cudaFreeHost(h_dataset);
	cudaFreeHost(h_distances);
	cudaFree(d_distances);
	return 0;
}

void printDistanceMatrix(float*h_distances, int numCities, int numFeatures)
{
	for (int i = 0; i < numCities; i++)
	{
		int city1Offset = i * numCities;
		for (int j = 0; j < numCities; j++)
		{
			printf("%f ", h_distances[city1Offset + j]);
		}
		printf("\n");
	}
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	// https://stackoverflow.com/a/14038590
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

double fRand(double fMin, double fMax)
{
	double f = (double) rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

vector<City> generateCities(int numCities, int gridDimX, int gridDimY)
{
	vector<City> cities;
	int id = 0;
	for (int i = 0; i < numCities; i++)
	{
		City city;
		city.id = id;
		city.x = fRand(0, gridDimX);
		city.y = fRand(0, gridDimY);
		cities.push_back(city);
		id++;
	}
	return cities;
}
