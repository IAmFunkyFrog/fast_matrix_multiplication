function calculate_avg_and_dev() {
    ((avg = 0))
    ((dev = 0))
    ((iter = $1))
    ((iters = $1))

    while [[ ${iter} -gt 0 ]] ; do
        res=$(./build/experiment -t -b $block_size -d $dims -a $2 -n)

        avg=$(echo "scale=6; $avg + $res" | bc | sed 's/^\./0./')
        dev=$(echo "scale=6; $dev + $res * $res" | bc | sed 's/^\./0./')
        ((iter -= 1))
    done

    avg=$(echo "scale=6; $avg / $iters" | bc | sed 's/^\./0./')
    dev=$(echo "scale=6; $dev / $iters - $avg * $avg + 0.000001" | bc | sed 's/^\./0./')
    echo "$avg,$dev"
}

# 32
make -s
((dims = 2880))
((diff = 256))
((iter = 1))
type=$1

echo "Dims,BlockSize,Avg,Dev"
while [[ ${iter} -gt 0 ]] ; do
    for block_size in 16 32 80 120 240; do
        res=$(calculate_avg_and_dev 3 $type)
        echo "$dims,$block_size,$res"
    done
    ((dims += diff))
    ((iter -= 1))
done
