#!/bin/bash
# Usage: ./attention_svm_pipeline.sh <outdir>
# Note: the `--force` flag always overwrites the output directory
# Note: edit the runcard before launching the script !

outdir=$1
runcard=../cards/runard.yaml

mkdir $outdir
quake datagen $runcard -o $outdir --force
quake train -o $outdir -m attention --force
quake train -o $outdir -m svm --force