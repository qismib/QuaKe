# min_en=("0.1" "0.05" "0.01")
min_en=("0.05")

# res=($(seq 0.5 0.25 7.5))
res=($(seq 0.5 0.25 7.5))


for j in "${!min_en[@]}"; do
    for i in "${!res[@]}"; do
    printf "${min_en[$j]}"
    printf "${res[$i]}"
        quake datagen cards/runcard.yaml -o ../output_provola --force --res ${res[$i]} --energy ${min_en[$j]}
        quake train -o ../output_provola -m blob --force
        # quake train -o ../output_provola -m attention --force
    done
done