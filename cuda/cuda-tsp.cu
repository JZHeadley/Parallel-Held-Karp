#include <stdio.h>
#include <algorithm>
#include <iterator>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>
#include <map>
using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

typedef struct
{
	int id;
	double x;
	double y;
} City;
typedef struct
{
	double cost;
	vector<int> path;
} PathCost;

void printDistanceMatrix(float*h_distances, int numCities, int numFeatures);
double fRand(double fMin, double fMax);
vector<City> generateCities(int numCities, int gridDimX, int gridDimY);
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true);
void genKey(vector<int> set, int z, long long &key);

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

__device__ unsigned long long countNumBits(unsigned long long n)
{
	unsigned long long count = 0;
	while (n)
	{
		count += n & 1;
		n >>= 1;
	}
	return count;
}

__device__ unsigned long long curPosition = 0;
__global__ void findPermutations(char* permutationsOfK, int k, unsigned long long lowerBound, unsigned long long upperBound)
{
	curPosition = 0;
	unsigned long long tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long long numToCheck = lowerBound + tid;
	unsigned long long count = 0;
	unsigned long long curBitPosition = 0;
	if (numToCheck < upperBound)
	{

		if (countNumBits(numToCheck) == k)
		{
			__syncthreads();
			unsigned long long added = atomicAdd(&curPosition, 1);
			if (k == 1)
			{
				printf("found a permutation %llu\n", added);
			}
			unsigned long long permutationStartPos = (added) * (unsigned long long) k;
			while (numToCheck)
			{
				if (numToCheck & 1)
				{
					permutationsOfK[permutationStartPos + count] = curBitPosition;
					count++;
				}
				numToCheck >>= 1;
				curBitPosition++;
			}

		}
	}
}

vector<City> tsp(vector<City> cities, int numCities, float* distances, float* d_distances)
{

	cudaEvent_t permutationsStart, permutationsStop;
	cudaEventCreate(&permutationsStart);
	cudaEventCreate(&permutationsStop);
	float permutationMilliseconds = 0;
	long long key = 0x00000;
	map<long long int, PathCost> solutionsMap;
	vector<int> cityNums;
	// convert cities back to integer array
	for (int i = 1; i < numCities; i++)
	{
		cityNums.push_back(i);
	}
	// calculate the highest layer number so we know how large we need to be for our permutation storage at worst
	int k = numCities % 2 == 0 ? numCities / 2 : (ceil(numCities / 2));
	// initalize first 2 levels of the lookup table
	for (int i = 1; i < numCities; i++)
	{
		for (int j = 1; j < numCities; j++)
		{
			if (i == j)
				continue;
			vector<int> iSet
			{ i };
			genKey(iSet, j, key);
			PathCost pathCost;
			vector<int> path
			{ 0, i };
			pathCost.path = path;
			pathCost.cost = distances[i * numCities + j] + distances[0 + i];
			solutionsMap.insert(pair<long long, PathCost>(key, pathCost));
		}
	}
	double currentCost = 0;
	char* d_permutationsOfK;
	char *h_permutationsOfK = (char*) malloc(pow(2, numCities) * sizeof(char) * k);
	gpuErrchk(cudaMalloc(&d_permutationsOfK, pow(2, numCities) * sizeof(char) * k));

	unsigned long long finalPos;
	unsigned long long numPossibilities = pow(2, numCities); // - pow(2, k - 1);
	int threadsPerBlock = 1024;
	unsigned long long blocksPerGrid = ((numPossibilities) + threadsPerBlock - 1) / threadsPerBlock;
	for (int subsetSize = 2; subsetSize < numCities; subsetSize++)
	{
		cudaEventRecord(permutationsStart);

		findPermutations<<<blocksPerGrid, threadsPerBlock, 0>>>(d_permutationsOfK, subsetSize, (unsigned long long) (pow(2, subsetSize) - 1),
				(unsigned long long) pow(2, numCities));
//		cudaDeviceSynchronize();
		gpuErrchk(cudaMemcpyFromSymbol(&finalPos, curPosition, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(h_permutationsOfK, d_permutationsOfK, finalPos * sizeof(char) * subsetSize, cudaMemcpyDeviceToHost));

		cudaEventRecord(permutationsStop);
		cudaEventSynchronize(permutationsStop);
		cudaEventElapsedTime(&permutationMilliseconds, permutationsStart, permutationsStop);
		printf("%i choose %i is %llu and took %f ms\n", numCities, subsetSize, finalPos, permutationMilliseconds);

		// use the permutations we generated here
		// remember the permutations are stored in k length 'arrays' within the 1-D array we have them in
		// so we need to index them interestingly.
		// converting to vector<vector<int>> so I don't have to rethink the logic at the current moment... definitely need to in the interests of speed
		vector<vector<int>> subsets;
		for (int pos = 0; pos < finalPos; pos++)
		{
			vector<int> permutation;
			for (int l = 0; l < subsetSize; l++)
			{
				permutation.push_back(h_permutationsOfK[pos * subsetSize + l]);
			}
			subsets.push_back(permutation);
		}
		int counter = 0;
		for (vector<int> set : subsets)
		{

			for (int k : set)
			{
				vector<int> kSet
				{ k };
				vector<int> diff;
				set_difference(set.begin(), set.end(), kSet.begin(), kSet.end(), inserter(diff, diff.begin()));
				double minCost = INT_MAX;
				vector<int> minPath;
				int bestM;
				counter++;
				// we initialized 2 levels earlier so this for loop will always be able to run.
				for (int m : diff)
				{
					vector<int> mSet
					{ m }; // need to generate the key for k-1
					vector<int> noMoreM; // get rid of m because thats where we're going
					set_difference(diff.begin(), diff.end(), mSet.begin(), mSet.end(), inserter(noMoreM, noMoreM.begin()));

					genKey(noMoreM, m, key);
					currentCost = solutionsMap[key].cost + distances[m * numCities + k];
					if (currentCost < minCost)
					{
						minCost = currentCost;
						minPath = solutionsMap[key].path;
						bestM = m;
					}
				}
				genKey(diff, k, key);

				PathCost pathCost;
				pathCost.cost = minCost;
				minPath.push_back(bestM);
				pathCost.path = minPath;
				solutionsMap.insert(pair<long long, PathCost>(key, pathCost));
			}
		}
		// printf("we have %i subsets of size %i\n", counter, i);
	}
	double minCost = INT_MAX;
	vector<int> minPath;
	int bestM;
	for (int m : cityNums)
	{
		vector<int> mSet
		{ m }; // need to generate the key for k-1
		vector<int> noMoreM; // get rid of m because thats where we're going
		set_difference(cityNums.begin(), cityNums.end(), mSet.begin(), mSet.end(), inserter(noMoreM, noMoreM.begin()));

		genKey(noMoreM, m, key);
		currentCost = solutionsMap[key].cost + distances[m * numCities + 0];
		if (currentCost < minCost)
		{
			minCost = currentCost;
			vector<int> path = solutionsMap[key].path;
			minPath = path;
			bestM = m;
		}
	}

	minPath.push_back(bestM);
	minPath.push_back(0);
	vector<City> bestPath;
	for (int i = 0; i < minPath.size(); i++)
	{
		bestPath.push_back(cities[minPath[i]]);
	}
	printf("Cost for this set of %i cities was %f\n", numCities, minCost);
	return bestPath;
}

int main(void)
{
	float* d_distances;
	float* h_distances;
	float* h_dataset;
	float* d_dataset;

	int numCities = 13;
	int numFeatures = 3;
	int k = numCities % 2 == 0 ? numCities / 2 : (ceil(numCities / 2));

	cudaEvent_t allStart, allStop, distStart, distStop;
	cudaEventCreate(&allStart);
	cudaEventCreate(&allStop);
	cudaEventCreate(&distStart);
	cudaEventCreate(&distStop);

	float allMilliseconds = 0, distMilliseconds = 0;

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

	gpuErrchk(cudaFree(d_dataset));

	vector<City> solution = tsp(cities, numCities, h_distances, d_distances);

//	threadsPerBlock = 1024;
//	int numPossibilities = pow(2, numCities); // - pow(2, k - 1);
//	blocksPerGrid = ((numPossibilities) + threadsPerBlock - 1) / threadsPerBlock;
//
//	gpuErrchk(cudaFree(d_dataset));
//	cudaEventRecord(permutationsStart);
//	gpuErrchk(cudaMalloc(&d_permutationsOfK, pow(2, numCities) * sizeof(char) * k));
//	char *h_permutationsOfK = (char*) malloc(pow(2, 29) * sizeof(char) * k);
//
//	for (int i = 1; i <= numCities; i++)
//	{
//		findPermutations<<<blocksPerGrid, threadsPerBlock, 0>>>(d_permutationsOfK, i, (unsigned long long) (pow(2, i) - 1),
//				(unsigned long long) pow(2, numCities));
//		unsigned long long finalPos;
//		cudaDeviceSynchronize();
//		gpuErrchk(cudaMemcpyFromSymbol(&finalPos, curPosition, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost));
////		finalPos++;
//		gpuErrchk(cudaMemcpy(h_permutationsOfK, d_permutationsOfK, finalPos * sizeof(char) * i, cudaMemcpyDeviceToHost));
//
//		printf("%i choose %i is %llu\n", numCities, i, finalPos);
////		printf("permutations for size %i\n", i);
////		for (int j = 0; j < finalPos; j++)
////		{
////			for (int z = 0; z < i; z++)
////			{
////				printf("%i\t", (int) h_permutationsOfK[j * i + z]);
////			}
////			printf("\n");
////		}
//
//	}
	gpuErrchk(cudaPeekAtLastError());
//	cudaDeviceSynchronize();

	cudaEventRecord(allStop);
	cudaEventSynchronize(allStop);
	cudaEventElapsedTime(&distMilliseconds, distStart, distStop);

	cudaEventElapsedTime(&allMilliseconds, allStart, allStop);

	printf("The distance calculation for %i cities took %llu ms.\n", numCities, (long long unsigned int) distMilliseconds);
//	printf("The permutations calculation for %i cities took %llu ms.\n", numCities, (long long unsigned int) permutationMilliseconds);
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
	for (int i = 0; i < numCities; i++)
	{
		City city;
		city.id = i;
		city.x = fRand(0, gridDimX);
		city.y = fRand(0, gridDimY);
		cities.push_back(city);
	}
//	City city0;
//	city0.id = 0;
//	city0.x = 323.05;
//	city0.y = 24.73;
//	cities.push_back(city0);
//	City city1;
//	city1.id = 1;
//	city1.x = 24.56;
//	city1.y = 101.00;
//	cities.push_back(city1);
//	City city2;
//	city2.id = 2;
//	city2.x = 275.87;
//	city2.y = 44.57;
//	cities.push_back(city2);
//	City city3;
//	city3.id = 3;
//	city3.x = 114.67;
//	city3.y = 186.45;
//	cities.push_back(city3);
//	City city4;
//	city4.id = 4;
//	city4.x = 164.11;
//	city4.y = 334.44;
//	cities.push_back(city4);
//	City city5;
//	city5.id = 5;
//	city5.x = 485.90;
//	city5.y = 401.21;
//	cities.push_back(city5);
//	City city6;
//	city6.id = 6;
//	city6.x = 333.49;
//	city6.y = 464.63;
//	cities.push_back(city6);
//	City city7;
//	city7.id = 7;
//	city7.x = 133.37;
//	city7.y = 168.05;
//	cities.push_back(city7);
//	City city8;
//	city8.id = 8;
//	city8.x = 362.79;
//	city8.y = 255.52;
//	cities.push_back(city8);
//	City city9;
//	city9.id = 9;
//	city9.x = 378.74;
//	city9.y = 235.48;
//	cities.push_back(city9);

	return cities;
}

void genKey(vector<int> set, int z, long long &key)
{
	key = 0;
	key |= z;
	for (int j : set)
	{
		key |= (1 << (j + 8));
	}
}
