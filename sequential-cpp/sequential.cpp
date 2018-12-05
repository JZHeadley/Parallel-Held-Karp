#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <map>
#include <string.h>
#include <limits.h>

#include "include/support.h"

using namespace std;

typedef struct
{
    double cost;
    vector<int> path;
} PathCost;

double **distances;
map<long long int, PathCost> solutionsMap;
vector<int> bestPath;
double bestCost;

double distance(City c1, City c2)
{
    return sqrt(pow(c1.x - c2.x, 2) + pow(c1.y - c2.y, 2));
}

vector<City> readCities(char *filePath)
{
    vector<City> cities;
    ifstream file(filePath);
    double x, y;
    int id = 0;
    // Reading in the cities tried to methodize this and failed
    while (file >> x >> y)
    {
        City city;
        city.id = id;
        city.x = x;
        city.y = y;
        cities.push_back(city);
        id++;
    }
    printf("We just read in %i cities\n", cities.size());
    return cities;
}

double **computeDistanceMatrix(vector<City> cities)
{
    double **distances = (double **)malloc(cities.size() * sizeof(double *));
    for (int i = 0; i < cities.size(); i++)
        distances[i] = (double *)malloc(cities.size() * sizeof(double));

    for (int i = 0; i < cities.size(); i++)
    {
        for (int j = 0; j < cities.size(); j++)
        {
            City city1 = cities[i];
            City city2 = cities[j];
            distances[i][j] = sqrt(pow(city1.x - city2.x, 2) + pow(city1.y - city2.y, 2));
        }
    }
    return distances;
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

vector<vector<int>> generateSubsets(int size, int n)
{
    int count = 0;
    vector<vector<int>> container;
    vector<int> row;
    vector<bool> v((unsigned long)n);
    fill(v.begin(), v.begin() + size, true);

    do
    {
        for (int i = 0; i < n; ++i)
        {
            if (v[i])
            {
                count++;
                row.push_back(i + 1);
                if (count == size)
                {
                    container.push_back(row);
                    row.clear();
                    count = 0;
                }
            }
        }
    } while (prev_permutation(v.begin(), v.end()));
    return container;
}

void tsp(vector<City> cities, int start, int numCities)
{
    long long key = 0x00000;
    vector<int> cityNums;
    // convert cities back to integer array
    for (int i = 1; i < numCities; i++)
    {
        cityNums.push_back(i);
    }

    // initalize first 2 levels of the lookup table
    for (int i = 1; i < numCities; i++)
    {
        for (int j = 1; j < numCities; j++)
        {
            if (i == j)
                continue;
            vector<int> iSet{i};
            genKey(iSet, j, key);
            PathCost pathCost;
            vector<int> path{0, i};
            pathCost.path = path;
            pathCost.cost = distances[i][j] + distances[0][i];
            solutionsMap.insert(pair<long long, PathCost>(key, pathCost));
        }
    }
    //we're good so far...
    double currentCost = 0;
    for (int i = 2; i < numCities; i++)
    { // iterate through all cardinalities of subsets
        // printf("working on subsets of size %i\n", i);
        vector<vector<int>> subsets = generateSubsets(i, cityNums.size());
        int counter = 0;
        for (vector<int> set : subsets)
        {

            for (int k : set)
            {
                vector<int> kSet{k};
                vector<int> diff;
                set_difference(set.begin(), set.end(), kSet.begin(), kSet.end(), inserter(diff, diff.begin()));
                double minCost = INT_MAX;
                vector<int> minPath;
                int bestM;
                    counter++;
                // we initialized 2 levels earlier so this for loop will always be able to run.
                for (int m : diff)
                {
                    vector<int> mSet{m}; // need to generate the key for k-1
                    vector<int> noMoreM; // get rid of m because thats where we're going
                    set_difference(diff.begin(), diff.end(), mSet.begin(), mSet.end(), inserter(noMoreM, noMoreM.begin()));

                    genKey(noMoreM, m, key);
                    currentCost = solutionsMap[key].cost + distances[m][k];
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
        printf("we have %i subsets of size %i\n", counter, i);
    }
    double minCost = INT_MAX;
    vector<int> minPath;
    int bestM;
    for (int m : cityNums)
    {
        vector<int> mSet{m}; // need to generate the key for k-1
        vector<int> noMoreM; // get rid of m because thats where we're going
        set_difference(cityNums.begin(), cityNums.end(), mSet.begin(), mSet.end(), inserter(noMoreM, noMoreM.begin()));

        genKey(noMoreM, m, key);
        currentCost = solutionsMap[key].cost + distances[m][0];
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
    bestPath = minPath;
    bestCost = minCost;
}
void printPath(vector<int> path)
{

    printf("path is: ");
    for (int i = 0; i < path.size() - 1; i++)
    {
        printf("%i -> ", path[i]);
    }
    printf("%i", path[path.size() - 1]);

    printf("\n");
}
int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage:  ./tsp <dataset path>\n");
        exit(1);
    }
    vector<City> cities = readCities(argv[1]);
    if (cities.size() == 0)
    {
        printf("Please use a dataset file that is not empty\n");
        exit(2);
    }

    for (City city : cities)
    {
        cout << "X is " << city.x << " Y is " << city.y << " for city with id " << city.id << endl;
    }

    distances = computeDistanceMatrix(cities);
    //printMatrix(distances, cities.size(), cities.size());

    int *cityIds = (int *)malloc(cities.size() * sizeof(int));
    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    tsp(cities, 0, cities.size());

    printPath(bestPath);
    printf("cost was %f\n", bestCost);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("TSP ran in %llu ms for %i cities\n", (long long unsigned int)diff, cities.size());
}
