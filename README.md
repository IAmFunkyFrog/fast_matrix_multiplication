```
Program for benchamrking of matrix multiplication
         A * B = C
where:
        A - upper triangular matrix
        B - square matrix

Options:
        -n - no verify, disable verification after run
        -d - set matrix dimension size (default 2880)
        -b - set matrix block size (default 2880 / 16)
        -s - set seed for random matrix fill
        -t - print elapsed time (only for parallel build!)
        -a - use one of 3 algorithms
                1 - simple matrix multiplication
                2 - save A as flat array and make blocked multiplication
                3 - save A and B as blocked matrices and make blocked multiplication
                4 - same as 2 but each block multiplied with OMP task on different cores
```
