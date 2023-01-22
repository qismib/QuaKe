min_en=("0.05") #,"0.01" "0.05" "0.1")
res=($(seq 1 0.25 7.5))


# quake train -o ../output -m blob --force
# quake train -o ../output -m attention --force
# quake train -o ../output -m cnn --force

for j in "${!min_en[@]}"; do
    for i in "${!res[@]}"; do
    printf "${min_en[$j]}"
    printf "${res[$i]}"
        quake datagen cards/runcard.yaml -o ../output_provola --force --res ${res[$i]} --energy ${min_en[$j]}
        quake train -o ../output_provola -m blob --force
        quake train -o ../output_provola -m attention --force
        quake train -o ../output_provola -m cnn --force
    done
done