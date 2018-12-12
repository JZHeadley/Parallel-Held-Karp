#!/bin/bash

for numCities in {10..25}
do
    `./genCities.sh $numCities > datasets/$numCities.cities`
    `./sequential-tsp datasets/$numCities.cities > results/results.$numCities &`
done
