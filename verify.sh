
make
echo "Verification of algorithm 2"
for seed in `seq 0 1 10`; do
    ./build/experiment -s $seed -a 2 -d 512
    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Test $seed failed"
    else
        echo "Test $seed passed: $retVal"
    fi
done

echo "Verification of algorithm 3"
for seed in `seq 0 1 10`; do
    ./build/experiment -s $seed -a 3 -d 512 -b 16
    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Test $seed failed"
    else
        echo "Test $seed passed: $retVal"
    fi
done

echo "Verification of algorithm 4"
for seed in `seq 0 1 10`; do
    ./build/experiment -s $seed -a 4 -d 512
    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Test $seed failed"
    else
        echo "Test $seed passed: $retVal"
    fi
done
