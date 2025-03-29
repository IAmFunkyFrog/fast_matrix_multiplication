#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include <time.h>
#include <omp.h>
#include <getopt.h>

#define DEFAULT_DIM_SIZE 2880
#define DEFAULT_BLOCK_SIZE (DEFAULT_DIM_SIZE / 16)

#define TIME_ME(CODE, time_var) \
    { \
        double start = omp_get_wtime();; \
        CODE; \
        double end = omp_get_wtime(); \
        time_var = end - start; \
    }

// Note: how much digits after . we use in comparison of float numbers
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

#define HELP \
    "Program for benchamrking of matrix multiplication\n" \
    "\t A * B = C\n" \
    "where:\n" \
    "\tA - upper triangular matrix\n" \
    "\tB - square matrix\n" \
    "\nOptions:\n" \
    "\t-n - no verify, disable verification after run\n" \
    "\t-d - set matrix dimension size (default 2880)\n" \
    "\t-b - set matrix block size (default 2880 / 16)\n" \
    "\t-s - set seed for random matrix fill\n" \
    "\t-t - print elapsed time (only for parallel build!)\n" \
    "\t-a - use one of 3 algorithms\n" \
    "\t\t1 - simple matrix multiplication\n" \
    "\t\t2 - save A as flat array and make blocked multiplication\n" \
    "\t\t3 - save A and B as blocked matrices and make blocked multiplication\n" \
    "\t\t4 - same as 2 but each block multiplied with OMP task on different cores\n"

int main(int argc, char *argv[]) {
    int dimension_size = DEFAULT_DIM_SIZE;
    int block_size = DEFAULT_BLOCK_SIZE;
    int random_seed = 42;
    int algorithm = 4;

    int print_elapsed_time = 0;
    int should_verify = 1;

    int opt;
    while ((opt = getopt(argc, argv, "d:b:s:ta:n")) != -1) {
        switch (opt) {
        case 'd':
            dimension_size = atoi(optarg);
            assert(dimension_size > 0);
            break;
        case 'b':
            block_size = atoi(optarg);
            assert(block_size > 0);
            break;
        case 's':
            random_seed = atoi(optarg);
            break;
        case 't':
            print_elapsed_time = 1;
            break;
        case 'a':
            algorithm = atoi(optarg);
            assert(algorithm >= 1 && algorithm <= 4);
            break;
        case 'n':
            should_verify = 0;
            break;
        default:
            fprintf(stderr, HELP);
            return -1;
        }
    }

    double_matrix_t A = matrix_allocate_upper_triangular_cols(dimension_size);
    double_matrix_t B = matrix_allocate(dimension_size, dimension_size);

    matrix_fill_random(A);
    matrix_fill_random(B);

    double time_in_seconds;
    double_matrix_t result = matrix_allocate(A.nrows, B.ncols);
    switch (algorithm) {
        default:
            fprintf(stderr, HELP);
            return -1;
        case 1: {
            double_matrix_t A_normal = matrix_convert_to_normal(A);
            double_matrix_t B_normal = matrix_convert_to_normal(B);
            TIME_ME(
                matrix_mult3(A_normal, B_normal, result),
                time_in_seconds
            );
            break;
        }
        case 2: {
            TIME_ME(
                matrix_mult_block3(A, B, result, block_size),
                time_in_seconds
            );
            break;
        }
        case 3: {
            double_matrix_t A_blocked = matrix_convert_to_upper_triangular_blocked(A, block_size);
            double_matrix_t B_blocked = matrix_convert_to_normal_blocked(B, block_size);
            TIME_ME(
                matrix_mult_block3(A_blocked, B_blocked, result, block_size),
                time_in_seconds
            );
            break;
        }
        case 4: {
            TIME_ME(
                matrix_omp_mult_block3(A, B, result, block_size),
                time_in_seconds
            );
            break;
        }
    }

    if (should_verify) {
        double_matrix_t A_normal = matrix_convert_to_normal(A);
        double_matrix_t B_normal = matrix_convert_to_normal(B);
        double_matrix_t expected = matrix_allocate(A_normal.nrows, B_normal.ncols);
        matrix_mult3(A_normal, B_normal, expected);
        if (!verify(expected, result)) {
            fprintf(stderr, "Verification failed!\n");
            return -1;
        }
    }

    if (print_elapsed_time)
        printf("%.3f\n", time_in_seconds);


/*
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
*/
    return 0;
}