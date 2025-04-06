mkdir -p results

echo "Experiment 3"
bash ./run_experiment_with_blocks.sh 3 > ./results/blocked_arrays_blocked_multiplication.csv
echo "Experiment 4"
bash ./run_experiment_with_blocks.sh 4 > ./results/flat_arrays_blocked_multiplication_omp.csv
echo "Experiment 1"
bash ./run_experiment.sh 1 > ./results/baseline.csv
echo "Experiment 2"
bash ./run_experiment_with_blocks.sh 2 > ./results/flat_arrays_blocked_multiplication.csv
