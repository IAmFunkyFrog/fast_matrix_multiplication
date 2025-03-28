#include "matrix.h"
#include <omp.h>
#include <stdio.h>

#define MIN(x, y) ((x < y) ? (x) : (y))

static double random_double() {
    union {
        double d;
        unsigned char uc[sizeof(double)];
    } u;

    do {
        for (unsigned i = 0; i < sizeof(u.uc); i++) {
            u.uc[i] = (unsigned char) rand();
        }
    } while (!isfinite(u.d));

    while (u.d > 10.0 || u.d < -10.0) u.d /= 10.0;

    return u.d;
}

static double matrix_get_or_zero(double_matrix_t matrix, int i, int j) {
    if (!matrix_index_in_matrix(matrix, i, j)) return 0;

    return matrix_get(matrix, i, j);
}

void matrix_fill_random(double_matrix_t matrix) {
    for (int i = 0; i < matrix.nrows; i++)
        for (int j = 0; j < matrix.ncols; j++)
            if (matrix_index_in_matrix(matrix, i, j))
                matrix_set(matrix, i, j, random_double());
}

#ifndef PARALLEL 

void matrix_mult3(double_matrix_t m1, double_matrix_t m2, double_matrix_t out) {
    assert(out.nrows == m1.nrows && out.ncols == m2.ncols);
    assert(m1.type == m2.type && m1.type == NORMAL);

    for (int i = 0; i < out.nrows; i++) {
        for (int j = 0; j < out.ncols; j++) {
            double val = 0.0;
            for (int k = 0; k < m2.nrows; k++)
                val += matrix_get(m1, i, k) * matrix_get(m2, k, j);
            matrix_set(out, i, j, val);
        }
    }
}

#else

void matrix_mult3(double_matrix_t m1, double_matrix_t m2, double_matrix_t out) {
    assert(out.nrows == m1.nrows && out.ncols == m2.ncols);

    int i, j, k;
    #pragma omp parallel for private(i, j, k)
    for (i = 0; i < out.nrows; i++) {
        for (j = 0; j < out.ncols; j++) {
            double val = 0.0;
            for (k = 0; k < m2.nrows; k++)
                val += matrix_get(m1, i, k) * matrix_get(m2, k, j);
            matrix_set(out, i, j, val);
        }
    }
}

#endif

void matrix_mult_block3(double_matrix_t m1, double_matrix_t m2, double_matrix_t out, int block_max_size) {
    assert(out.nrows == m1.nrows && out.ncols == m2.ncols && m1.ncols == m2.nrows);
    assert(m1.type == m2.type && m1.type == NORMAL);

    for (int block_start_i = 0; block_start_i < out.nrows; block_start_i += block_max_size) {
        for (int block_start_j = 0; block_start_j < out.ncols; block_start_j += block_max_size) {
            for (int block_start_k = 0; block_start_k < m2.nrows; block_start_k += block_max_size) {
                for (int i = block_start_i; i < MIN(out.nrows, block_start_i + block_max_size); i++) {
                    for (int j = block_start_j; j < MIN(out.ncols, block_start_j + block_max_size); j++) {
                        double val = matrix_get(out, i, j);
                            for (int k = block_start_k; k < MIN(m2.nrows, block_start_k + block_max_size); k++)
                                val += matrix_get(m1, i, k) * matrix_get(m2, k, j);
                        matrix_set(out, i, j, val);
                    }
                }
            }
        }
    }

    return;
}
