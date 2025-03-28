#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include <time.h>

#define DEFAULT_DIM_SIZE 2880
#define DEFAULT_BLOCK_SIZE (DEFAULT_DIM_SIZE / 16)

#define TIME_ME(CODE) \
    { \
        clock_t start = clock(); \
        CODE; \
        clock_t end = clock(); \
        float seconds = (float)(end - start) / CLOCKS_PER_SEC; \
        printf("Code time %.3f in seconds\n", seconds); \
    }

int main(int argc, char *argv[]) {
    int dimension_size = DEFAULT_DIM_SIZE;
    int block_size = DEFAULT_BLOCK_SIZE;
    double_matrix_t A = matrix_allocate_upper_triangular_cols(dimension_size);
    double_matrix_t B = matrix_allocate_lower_triangular_cols(dimension_size);

    matrix_fill_random(A);
    matrix_fill_random(B);

    {
        double_matrix_t A_normal = matrix_convert_to_normal(A);
        double_matrix_t B_normal = matrix_convert_to_normal(B);
        double_matrix_t out1 = matrix_allocate(A_normal.nrows, B_normal.ncols);
        double_matrix_t out2 = matrix_allocate(A_normal.nrows, B_normal.ncols);

        printf("Normal, no block:\n");
        TIME_ME(
            matrix_mult3(A_normal, B_normal, out1)
        );

        printf("Normal, block:\n");
        TIME_ME(
            matrix_mult_block3(A_normal, B_normal, out2, block_size)
        );

        for (int i = 0; i < out1.nrows; i++)
            for (int j = 0; j < out1.ncols; j++)
                if (((int) matrix_get(out1, i, j) * 100) != ((int) matrix_get(out2, i, j) * 100))
                    printf("MATRIX NOT SAME: %.3f != %.3f\n", matrix_get(out1, i, j), matrix_get(out2, i, j));
    }

    return 0;
}