#!/bin/bash

for numThreads in 512 1024 2048 4096
do
    for numCities in {10..25}
    do
        ./cuda-tsp $numCities $numThreads | tee results/results.$numCities.$numThreads
    done
done
