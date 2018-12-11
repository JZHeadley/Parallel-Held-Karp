#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <math.h>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <iterator>
#include <limits.h>
#include <float.h>
#include <chrono>
#include <thread>
#include <mutex>
using Clock = std::chrono::steady_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using namespace std;

#define NUM_THREADS 4000

struct City
{
	int id;
	double x;
	double y;
};
struct Salesman
{
	double minCost;
	vector<int> minPath;
};

struct TravelArgs
{
	int threadId;
};

//support as many nodes as possible even though it would take years
map <long long, Salesman> shortPath;
double **dMatrix;
mutex m;
vector<vector<int>> subsets;

double distance(City city1, City city2)
{
	return sqrt((pow((city2.x - city1.x),2)) + (pow((city2.y - city1.y),2)));
}

void distanceMatrix(vector<City> loc)
{
	int i, j, size = loc.size();
	dMatrix = (double **)malloc(size * sizeof(double)); 
	for(i = 0; i < size; i++)
	{
		dMatrix[i] = (double *)malloc(size * sizeof(double));
	}

	for(i = 0; i < size; i++)
	{
		for(j = 0; j < size; j++)
		{
			dMatrix[i][j] = distance(loc[i], loc[j]);
		}
	}
}

vector<vector<int> > generateSubsets(int size, int n)
{
	int count = 0;
	vector<vector<int> > allSets;
	vector<int> set;
	vector<bool> v((unsigned long)n);
	fill(v.begin(), v.begin() + size, true);
	do {
		for (int i = 0; i < n; ++i) {
			if (v[i]) {
				count++;
				set.push_back(i + 1);
				if (count == size) {
					allSets.push_back(set);
					set.clear();
					count = 0;
				}
			}
		}
	} while (prev_permutation(v.begin(), v.end()));

	return allSets;
}

void genKey(vector<int> &set, int z, long long &key)
{
	//we set our key up in a way that the lowest byte is the z in C(z,S) and the remaining 56 bits are a mask
	//where if set S = {1,2,3} bits 1, 2, and 3 are 1
	key = 0;
	key |= z;
	for(int j = 0; j < set.size(); j++)
	{
		key |= (1 << set[j]+8);
	}
}

void* parallel(void * id)
{
	int threadId = *(int *) id;
	int minM;
	double minCost;
	double currentCost;
	vector<int> kSet;
	vector<int> minPath;
	vector<int> setWithoutK;
	vector<int> mSet;
	vector<int> setWithoutM;
	vector<int> currentSet;
	long long key;
	int instances = subsets.size();
	int size;
	int kSize;

	int instancesPerTask = (instances + NUM_THREADS) / NUM_THREADS;
    	int beginIndex = threadId * instancesPerTask;
	int endIndex = min((threadId + 1) * instancesPerTask, instances);

	for(int i = beginIndex; i < endIndex; i++)
	{
		currentSet = subsets[i];
		size = currentSet.size();
		kSize = size - 1;
		for(int k = 0; k < size; k++) //in order to find C(z,S) we need to find the min C(k,{S}-{k}) for all k
		{
			kSet.clear();
			kSet.push_back(currentSet[k]);
			setWithoutK.clear();
			set_difference(currentSet.begin(), currentSet.end(), kSet.begin(), kSet.end(), inserter(setWithoutK, setWithoutK.begin()));
			minCost = DBL_MAX;
			minM = 0;
			for(int m = 0; m < kSize; m++) //find min C(m, S-{k}) + dMatrix[m][k]
			{	
				mSet.clear();
				mSet.push_back(setWithoutK[m]);
				setWithoutM.clear();	
				set_difference(setWithoutK.begin(), setWithoutK.end(), mSet.begin(), mSet.end(), inserter(setWithoutM, setWithoutM.begin()));

				genKey(setWithoutM, setWithoutK[m], key);
				currentCost = shortPath[key].minCost + dMatrix[setWithoutK[m]][currentSet[k]];
				if(currentCost < minCost)
				{
					minCost = currentCost;
					minPath = shortPath[key].minPath;
					minM = m;
				}
			}
			//add the path to the min path for this route
			minPath.push_back(setWithoutK[minM]);
			// now store the answer for the next iteration 
			Salesman s;
			s.minCost = minCost;
			s.minPath = minPath;
			genKey(setWithoutK, currentSet[k], key);
			m.lock();
			shortPath.insert(pair <long long, Salesman> (key, s));
			m.unlock();
		}
	}
	return NULL;
}

//unsigned long long maxThreads(int size)
//{
//	int maxLayer = ceil((double)size / 2.0);
//return binomialCoeff(size - 2, maxLayer - 1)
//}

// Returns value of Binomial Coefficient C(n, k) 
int binomialCoeff(int n, int k) //this only works up to about 30
{ 
	unsigned long long res = 1; 

	// Since C(n, k) = C(n, n-k) 
	if ( k > n - k ) 
		k = n - k; 

	// Calculate value of [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1] 
	for (int i = 0; i < k; ++i) 
	{ 
		res *= (n - i); 
		res /= (i + 1); 
	} 

	return res; 
}

void travel(int size)
{
	long long key;
	double currentCost = 0;
	vector<int> minPath;
	double minCost = DBL_MAX;
	int minM = 0;	
	vector<int> allCities;
	vector<int> setWithoutM;
	vector<int> mSet;
	vector<int> currentSet;
	int subsetSize;
	vector<pthread_t> threads;
	int instancesPerTask;
	int beginIndex;
	int endIndex;
	int *ids = (int *) malloc(NUM_THREADS * sizeof(int));
	int runs;


	if(size == 0)
	{
		minCost = 0;
		minPath.push_back(0);
	}
	else if(size == 1)
	{
		minCost = 0;
		minPath.push_back(0);
	}
	// find C(1,{}), C(2,{}) = dMatrix[0][1] dMatrix[0][2] etc.
	// and the ones with set cardnality = 1. these are special cases for our dynamic approach
	for(int i = 1; i < size; i++)
	{
		for(int j = 1; j < size; j++) 
		{
			if(i == j)
			{
				continue;
			}
			vector<int> iSet {i};
			genKey(iSet, j, key);
			Salesman s;
			s.minCost = dMatrix[i][j] + dMatrix[0][i];
			vector<int> minPath{0,i};
			s.minPath = minPath;
			shortPath.insert(pair <long long, Salesman> (key, s));
		}	
	}

	for(int i = 2; i < size; i++)//from subset size 2 to n-1
	{
		threads.clear();
		subsets = generateSubsets(i, size-1);
		subsetSize = subsets.size();
		runs = min(subsetSize, NUM_THREADS); 
			
		for(int j = 0; j < runs; j++)
		{
			ids[j] = j;
		}
		
		for(int j = 0; j < runs; j++)
		{
			pthread_t thread;
			pthread_create(&thread, NULL, parallel, (void *)&ids[j]);
			threads.push_back(thread);
		}

		for(int j = 0; j < runs; j++)
		{
			pthread_join(threads[j],NULL);
		}
	}

	minCost = DBL_MAX;
	for(int m = 1; m < size; m++)
	{
		allCities.push_back(m);
	}
	for(int m = 0; m < allCities.size(); m++)
	{
		mSet.clear();
		mSet.push_back(allCities[m]);
		setWithoutM.clear();	
		set_difference(allCities.begin(), allCities.end(), mSet.begin(), mSet.end(), inserter(setWithoutM, setWithoutM.begin()));

		genKey(setWithoutM, allCities[m], key);
		currentCost = shortPath[key].minCost + dMatrix[allCities[m]][0];
		if(currentCost < minCost)
		{
			minCost = currentCost;
			minPath = shortPath[key].minPath;
			minM = m;
		}
	}
	//add the path to the min path for this route
	minPath.push_back(allCities[minM]);
	minPath.push_back(0);
	printf("MinPath:\n");
	for(int q = 0; q < minPath.size(); q++)
	{
		printf("%i,",minPath[q]);
	}	
	printf("\n");
	printf("MinCost: %f\n",minCost);
}

int main(int argc, char* argv[])
{
	vector<City> cities;	
	double x, y, id = 0;
	string filename = "cities.txt";
	ifstream infile(filename);
	while (infile >> x >> y)
	{
		City city;
		city.x = x;
		city.y = y;
		city.id = id;
		cities.push_back(city);
		id++;
	}
	int size = cities.size();
	distanceMatrix(cities);

	time_point<Clock> before = Clock::now();
	travel(size);
	time_point<Clock> after = Clock::now();
	milliseconds total = duration_cast<milliseconds>(after - before);
	int msec = total.count();
	printf("Total time elapsed for tsp: %d milliseconds\n",msec);
	return 0;
}
