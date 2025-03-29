#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>

// Note: fro blocked types, each block in matrix is simply
// NORMAL matrix, and normal matrix in this library is plain
// array with columns of matrix
typedef enum {
    NORMAL,
    UPPER_TRIANGULAR_COLS,
    NORMAL_BLOCKED,
    UPPER_TRIANGULAR_BLOCKED
} matrix_type_t;

typedef struct {
    int block_size;
    int blocks_in_row;
} matrix_type_info_blocked_t;

typedef union {
    matrix_type_info_blocked_t blocked_info; // for NORMAL_BLOCKED and UPPER_TRIANGULAR_BLOCKED
} matrix_type_info_t;

typedef struct {
    matrix_type_t type;
    matrix_type_info_t minfo;
    int ncols;
    int nrows;
    void *data;
} double_matrix_t;

static inline int matrix_index_in_matrix(double_matrix_t matrix, int i, int j) {
    if (i >= matrix.nrows || i < 0) return 0;
    if (j >= matrix.ncols || j < 0) return 0;

    if (matrix.type == NORMAL || matrix.type == NORMAL_BLOCKED) return 1;
    if ((matrix.type == UPPER_TRIANGULAR_COLS || matrix.type == UPPER_TRIANGULAR_BLOCKED) && i <= j) return 1;

    return 0;
}

// TODO unify with matrix_get
static inline double matrix_get_UPPER_TRIANGULAR(double_matrix_t matrix, int i, int j) {
    double *plain = (double *) matrix.data;
    return plain[j * (j + 1) / 2 + i];
}

// TODO unify with matrix_get
static inline double matrix_get_NORMAL(double_matrix_t matrix, int i, int j) {
    double *plain = (double *) matrix.data;
    return plain[i * matrix.ncols + j];
}

// TODO unify with matrix_set
static inline double matrix_add_NORMAL(double_matrix_t matrix, int i, int j, double val) {
    double *plain = (double *) matrix.data;
    #pragma omp atomic
    plain[i * matrix.ncols + j] += val;
}

static inline double matrix_get(double_matrix_t matrix, int i, int j) {
    assert (i >= 0 && i < matrix.nrows && j >= 0 && j <= matrix.ncols);
    assert (matrix_index_in_matrix(matrix, i, j));
    double *plain = (double *) matrix.data;

    switch (matrix.type)
    {
    case NORMAL:
        return plain[i * matrix.ncols + j];
    case UPPER_TRIANGULAR_COLS:
        return plain[j * (j + 1) / 2 + i];
    case UPPER_TRIANGULAR_BLOCKED:
    case NORMAL_BLOCKED: {
        int block_size = matrix.minfo.blocked_info.block_size;
        int blocks_in_row = matrix.minfo.blocked_info.blocks_in_row;
        int block_i = i / block_size;
        int block_j = j / block_size;
        double *plain_block = plain + (block_i * blocks_in_row + block_j) * (block_size * block_size);
        return matrix_get(
            (double_matrix_t) {
                .type = NORMAL,
                .ncols = block_size,
                .nrows = block_size,
                .data = plain_block
            },
            i % block_size,
            j % block_size
        );
    }
    default:
        assert(0 && "unsupported");
    }
}

static double matrix_get_or_zero(double_matrix_t matrix, int i, int j) {
    if (!matrix_index_in_matrix(matrix, i, j)) return 0;

    return matrix_get(matrix, i, j);
}

static inline void matrix_set(double_matrix_t matrix, int i, int j, double value) {
    assert (i >= 0 && i < matrix.nrows && j >= 0 && j <= matrix.ncols);
    assert (matrix_index_in_matrix(matrix, i, j));
    double *plain = (double *) matrix.data;

    switch (matrix.type)
    {
    case NORMAL:
        plain[i * matrix.ncols + j] = value;
        return;
    case UPPER_TRIANGULAR_COLS:
        plain[j * (j + 1) / 2 + i] = value;
        return;
    case UPPER_TRIANGULAR_BLOCKED:
    case NORMAL_BLOCKED: {
        int block_size = matrix.minfo.blocked_info.block_size;
        assert(block_size != 0);
        int blocks_in_row = matrix.minfo.blocked_info.blocks_in_row;
        assert(blocks_in_row != 0);
        int block_i = i / block_size;
        int block_j = j / block_size;
        double *plain_block = plain + (block_i * blocks_in_row + block_j) * (block_size * block_size);
        matrix_set(
            (double_matrix_t) {
                .type = NORMAL,
                .ncols = block_size,
                .nrows = block_size,
                .data = plain_block
            },
            i % block_size,
            j % block_size,
            value
        );
        return;
    }
    default:
        assert(0 && "unsupported");
    }
}

static inline double_matrix_t matrix_allocate(int nrows, int ncols) {
    assert(ncols > 0 && nrows > 0);
    return (double_matrix_t) {
        .type = NORMAL,
        .ncols = ncols,
        .nrows = nrows,
        .data = calloc(1, sizeof(double) * ncols * nrows)
    };
}

static inline double_matrix_t matrix_allocate_blocked(int dims, int block_size) {
    assert(dims > 0);
    assert(dims % block_size == 0 && "could not divide matrix on such blocks");
    return (double_matrix_t) {
        .type = NORMAL_BLOCKED,
        .minfo = (matrix_type_info_t) {
            (matrix_type_info_blocked_t) {
                .block_size = block_size,
                .blocks_in_row = dims / block_size
            }
        },
        .ncols = dims,
        .nrows = dims,
        .data = calloc(1, sizeof(double) * dims * dims)
    };
}

static inline double_matrix_t matrix_allocate_upper_triangular_cols(int dims) {
    assert(dims > 0);
    return (double_matrix_t) {
        .type = UPPER_TRIANGULAR_COLS,
        .ncols = dims,
        .nrows = dims,
        .data = calloc(1, (sizeof(double) / 2) * (1 + dims) * dims)
    };
}

static inline double_matrix_t matrix_allocate_upper_triangular_blocked(int dims, int block_size) {
    assert(dims > 0);
    return (double_matrix_t) {
        .type = UPPER_TRIANGULAR_BLOCKED,
        .minfo = (matrix_type_info_t) { 
            (matrix_type_info_blocked_t) {
                .block_size = block_size,
                .blocks_in_row = dims / block_size
            }
        },
        .ncols = dims,
        .nrows = dims,
        // Note: we can use less memory because very much blocks will be zeroed, but for
        // tests its okay
        .data = calloc(1, sizeof(double) * dims * dims)
    };
}

static inline void matrix_free(double_matrix_t matrix) {
    free(matrix.data);
}

static void matrix_convert(double_matrix_t m, double_matrix_t out) {
    assert(m.nrows == m.ncols && "only square matrices supported");
    for (int i = 0; i < m.nrows; i++)
        for (int j = 0; j < m.nrows; j++) {
            if (matrix_index_in_matrix(m, i, j) && matrix_index_in_matrix(out, i, j))
                matrix_set(out, i, j, matrix_get(m, i, j));
        }
}

static double_matrix_t matrix_convert_to_normal(double_matrix_t m) {
    double_matrix_t out = matrix_allocate(m.nrows, m.ncols);
    matrix_convert(m, out);
    return out;
}

static double_matrix_t matrix_convert_to_upper_triangular_cols(double_matrix_t m) {
    assert(m.nrows == m.ncols && "only square matrices supported");
    double_matrix_t out = matrix_allocate_upper_triangular_cols(m.nrows);
    matrix_convert(m, out);
    return out;
}

static double_matrix_t matrix_convert_to_normal_blocked(double_matrix_t m, int block_size) {
    assert(m.nrows == m.ncols && "supported only for squared matrices");
    double_matrix_t out = matrix_allocate_blocked(m.nrows, block_size);
    matrix_convert(m, out);
    return out;
}

static double_matrix_t matrix_convert_to_upper_triangular_blocked(double_matrix_t m, int block_size) {
    assert(m.nrows == m.ncols && "supported only for squared matrices");
    double_matrix_t out = matrix_allocate_upper_triangular_blocked(m.nrows, block_size);
    matrix_convert(m, out);
    return out;
}

void matrix_fill_random(double_matrix_t matrix);

void matrix_mult3(double_matrix_t m1, double_matrix_t m2, double_matrix_t out);
void matrix_mult_block3(double_matrix_t m1, double_matrix_t m2, double_matrix_t out, int block_max_size);
void matrix_omp_mult_block3(double_matrix_t m1, double_matrix_t m2, double_matrix_t out, int block_max_size);

static inline double_matrix_t matrix_mult2(double_matrix_t m1, double_matrix_t m2) {
    double_matrix_t out = matrix_allocate(m1.nrows, m2.ncols);
    matrix_mult3(m1, m2, out);
    return out;
}

static inline double_matrix_t matrix_mult_block2(double_matrix_t m1, double_matrix_t m2, int block_max_size) {
    double_matrix_t out = matrix_allocate(m1.nrows, m2.ncols);
    matrix_mult_block3(m1, m2, out, block_max_size);
    return out;
}
