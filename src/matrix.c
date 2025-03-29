#include "matrix.h"
#include <omp.h>
#include <stdio.h>

#define MIN(x, y) ((x < y) ? (x) : (y))
#define MAX(x, y) ((x < y) ? (y) : (x))

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

void matrix_fill_random(double_matrix_t matrix) {
    for (int i = 0; i < matrix.nrows; i++)
        for (int j = 0; j < matrix.ncols; j++)
            if (matrix_index_in_matrix(matrix, i, j))
                matrix_set(matrix, i, j, random_double());
}

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

// TODO we can make better specilazation: make matrix_get/set more optimized
void matrix_mult_block3_UPPER_TRIANGULAR_COLS_NORMAL_specialization(
    double_matrix_t m1, double_matrix_t m2, double_matrix_t out, int block_max_size
) {
    assert(m1.type == UPPER_TRIANGULAR_COLS && m2.type == NORMAL);

    for (int block_start_i = 0; block_start_i < out.nrows; block_start_i += block_max_size) {
        for (int block_start_j = 0; block_start_j < out.ncols; block_start_j += block_max_size) {
            for (int block_start_k = 0; block_start_k < m2.nrows; block_start_k += block_max_size) {
                for (int i = block_start_i; i < MIN(out.nrows, block_start_i + block_max_size); i++) {
                    for (int j = block_start_j; j < MIN(out.ncols, block_start_j + block_max_size); j++) {
                        double val = matrix_get(out, i, j);
                        // Note: use k = MAX(i, block_start_k) because m1 is UPPER TRIANGULAR matrix,
                        // and all k < i are 0
                        // TODO might be optimized more if we simply skip blocks were
                        // MAX(i, block_start_k) >= MIN(m2.nrows, block_start_k + block_max_size)
                        for (int k = MAX(i, block_start_k); k < MIN(m2.nrows, block_start_k + block_max_size); k++)
                            val += matrix_get_UPPER_TRIANGULAR(m1, i, k) * matrix_get_NORMAL(m2, k, j);
                        matrix_set(out, i, j, val);
                    }
                }
            }
        }
    }
}

void matrix_omp_mult_block3_UPPER_TRIANGULAR_COLS_NORMAL_specialization(
    double_matrix_t m1, double_matrix_t m2, double_matrix_t out, int block_max_size
) {
    assert(m1.type == UPPER_TRIANGULAR_COLS && m2.type == NORMAL && out.type == NORMAL);

    #pragma omp parallel
    #pragma omp single
    {
    for (int block_start_i = 0; block_start_i < out.nrows; block_start_i += block_max_size) {
        for (int block_start_j = 0; block_start_j < out.ncols; block_start_j += block_max_size) {
            for (int block_start_k = 0; block_start_k < m2.nrows; block_start_k += block_max_size) {
                #pragma omp task
                #pragma omp private(block_start_i, block_start_j, block_start_k)
                {
                for (int i = block_start_i; i < MIN(out.nrows, block_start_i + block_max_size); i++) {
                    for (int j = block_start_j; j < MIN(out.ncols, block_start_j + block_max_size); j++) {
                        double val = 0;
                        // Note: use k = MAX(i, block_start_k) because m1 is UPPER TRIANGULAR matrix,
                        // and all k < i are 0
                        // TODO might be optimized more if we simply skip blocks were
                        // MAX(i, block_start_k) >= MIN(m2.nrows, block_start_k + block_max_size)
                        for (int k = MAX(i, block_start_k); k < MIN(m2.nrows, block_start_k + block_max_size); k++)
                            val += matrix_get_UPPER_TRIANGULAR(m1, i, k) * matrix_get_NORMAL(m2, k, j);
                        matrix_add_NORMAL(out, i, j, val);
                    }
                }
                }
            }
        }
    }
        #pragma omp taskwait
    }
}

void matrix_mult_block3_no_specialization(
    double_matrix_t m1, double_matrix_t m2, double_matrix_t out, int block_max_size
) {
    for (int block_start_i = 0; block_start_i < out.nrows; block_start_i += block_max_size) {
        for (int block_start_j = 0; block_start_j < out.ncols; block_start_j += block_max_size) {
            for (int block_start_k = 0; block_start_k < m2.nrows; block_start_k += block_max_size) {
                for (int i = block_start_i; i < MIN(out.nrows, block_start_i + block_max_size); i++) {
                    for (int j = block_start_j; j < MIN(out.ncols, block_start_j + block_max_size); j++) {
                        double val = matrix_get_or_zero(out, i, j);
                            for (int k = block_start_k; k < MIN(m2.nrows, block_start_k + block_max_size); k++)
                                val += matrix_get_or_zero(m1, i, k) * matrix_get_or_zero(m2, k, j);
                        matrix_set(out, i, j, val);
                    }
                }
            }
        }
    }

    return;
}

void matrix_mult_block3(double_matrix_t m1, double_matrix_t m2, double_matrix_t out, int block_max_size) {
    assert(out.nrows == m1.nrows && out.ncols == m2.ncols && m1.ncols == m2.nrows);
    // Note: for now tested only square matrices, not squared multiplication
    // might fail
    assert(m1.ncols == m1.nrows && m2.ncols == m2.nrows);
    
    // FIXME rewrite with X macro
    if (m1.type == UPPER_TRIANGULAR_COLS && m2.type == NORMAL) {
        matrix_mult_block3_UPPER_TRIANGULAR_COLS_NORMAL_specialization(m1, m2, out, block_max_size);
    } else {
        fprintf(stderr, "Unknown specializtion for matrices of given types\n");
        matrix_mult_block3_no_specialization(m1, m2, out, block_max_size);
    }
}

void matrix_omp_mult_block3(double_matrix_t m1, double_matrix_t m2, double_matrix_t out, int block_max_size) {
    assert(out.nrows == m1.nrows && out.ncols == m2.ncols && m1.ncols == m2.nrows);
    // Note: for now tested only square matrices, not squared multiplication
    // might fail
    assert(m1.ncols == m1.nrows && m2.ncols == m2.nrows);
    
    // FIXME rewrite with X macro
    if (m1.type == UPPER_TRIANGULAR_COLS && m2.type == NORMAL) {
        matrix_omp_mult_block3_UPPER_TRIANGULAR_COLS_NORMAL_specialization(m1, m2, out, block_max_size);
    } else {
        fprintf(stderr, "Unknown how to multiply matrices of given types\n");
        exit(1);
    }
}
