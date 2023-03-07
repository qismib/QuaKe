min_en=("0.01" "0.05" "0.1", "0.2")
min_en=("0.01")
res=($(seq 0.5 0.25 7.5))
# res=("0.5")
# res=($(seq 3.75 0.25 7.5))
# res=("6.25")

for j in "${!min_en[@]}"; do
    for i in "${!res[@]}"; do
    printf "${min_en[$j]}"
    printf "${res[$i]}"
        quake datagen ../cards/runcard.yaml -o ../../output --force --res ${res[$i]} --energy ${min_en[$j]}
        # quake train -o ../../output -m blob --force
        quake train -o ../../output -m attention --force
        # quake train -o ../../output -m cnn --force
    done
done