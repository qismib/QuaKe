#!/bin/bash
# Usage: ./cnn_svm_pipeline.sh <outdir>
# Note: the `--force` flag always overwrites the output directory
# Note: edit the runcard before launching the script !

outdir=$1
# runcard=../cards/runcard.yaml
runcard=cards/runcard.yaml

# mkdir $outdir
quake datagen $runcard -o $outdir --force
quake train -o $outdir -m cnn --force
quake train -o $outdir -m svm --force