# Parallel Held Karp
The Sequential version of the Held-Karp algorithm takes ![O(2^n * sqrt(n)](https://latex.codecogs.com/svg.latex?%24O%282%5En%5Csqrt%7B2%7D%29%24) our parallel implementation should take something around ![O((n^2)*log(n))](https://latex.codecogs.com/svg.latex?%24O%28n%5E2%5Clog%28%7Bn%7D%29%29%24) by performing the calculations for each 'layer' of the traveling salesman graph in parallel.