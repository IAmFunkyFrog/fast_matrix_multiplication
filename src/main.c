#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include <time.h>
#include <omp.h>

#define DEFAULT_DIM_SIZE 2880
#define DEFAULT_BLOCK_SIZE (DEFAULT_DIM_SIZE / 16)

#define TIME_ME(CODE) \
    { \
        double start = omp_get_wtime();; \
        CODE; \
        double end = omp_get_wtime(); \
        double seconds = end - start; \
        printf("Code time %.3f in seconds\n", seconds); \
    }

#define PRECISION 1000000ll

int verify(double_matrix_t expected, double_matrix_t result) {
    for (int i = 0; i < result.nrows; i++)
        for (int j = 0; j < result.ncols; j++)
            if (((long long) matrix_get_or_zero(expected, i, j) * PRECISION) != ((long long) matrix_get_or_zero(result, i, j) * PRECISION)) {
                printf("MATRIX NOT SAME: %.3f != %.3f\n", matrix_get_or_zero(expected, i, j), matrix_get_or_zero(result, i, j));
                return 0;
            }
    return 1;
}

int print(double_matrix_t matrix) {
    for (int i = 0; i < matrix.nrows; i++) {
        for (int j = 0; j < matrix.ncols; j++)
            printf("%0.3f ", matrix_get_or_zero(matrix, i, j));
        printf("\n");
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int dimension_size = DEFAULT_DIM_SIZE;
    int block_size = DEFAULT_BLOCK_SIZE;
    double_matrix_t A = matrix_allocate_upper_triangular_cols(dimension_size);
    double_matrix_t B = matrix_allocate(dimension_size, dimension_size);

    matrix_fill_random(A);
    matrix_fill_random(B);

    {
        double_matrix_t A_normal = matrix_convert_to_normal(A);
        double_matrix_t B_normal = matrix_convert_to_normal(B);
        double_matrix_t A_blocked = matrix_convert_to_upper_triangular_blocked(A, block_size);
        double_matrix_t B_blocked = matrix_convert_to_normal_blocked(B, block_size);
        double_matrix_t out1 = matrix_allocate(A_normal.nrows, B_normal.ncols);
        double_matrix_t out2 = matrix_allocate(A_normal.nrows, B_normal.ncols);
        double_matrix_t out3 = matrix_allocate(A_normal.nrows, B_normal.ncols);
        double_matrix_t out4 = matrix_allocate(A_normal.nrows, B_normal.ncols);
        double_matrix_t out5 = matrix_allocate(A_normal.nrows, B_normal.ncols);

        printf("Upper triangular on normal, OMP, block:\n");
        TIME_ME(
            matrix_omp_mult_block3(A, B, out4, block_size)
        );

        printf("Blocked matrices multiplication, block:\n");
        TIME_ME(
            matrix_mult_block3(A_blocked, B_blocked, out5, block_size)
        );

        printf("Upper triangular on normal, block:\n");
        TIME_ME(
            matrix_mult_block3(A, B, out3, block_size)
        );

        printf("Normal, no block:\n");
        TIME_ME(
            matrix_mult3(A_normal, B_normal, out1)
        );

        printf("Normal, block:\n");
        TIME_ME(
            matrix_mult_block3(A_normal, B_normal, out2, block_size)
        );

        if (!verify(out1, out2))
            printf("Block multiplication failed\n");

        if (!verify(out1, out3))
            printf("Block multiplication for upper triangular failed failed\n");

        if (!verify(out1, out4))
            printf("OMP failed\n");
        
        if (!verify(out1, out5))
            printf("Blocked matrices failed\n");
    }

    return 0;
}