#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int id;
    double x;
    double y;
} City;

void printMatrix(double** matrix, int r, int c)
{

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            // printf("distance between city i%i and j%i is %f\n", i, j, matrix[i][j]);
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
}
