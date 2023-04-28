#!/bin/bash
# Usage: ./examples/attention_svm_pipeline.sh <outdir>
# Note: the `--force` flag always overwrites the output directory
# Note: edit the runcard before launching the script !

outdir=$1
runcard=cards/runcard.yaml

deeplar datagen $runcard -o $outdir --force
deeplar train -o $outdir -m attention --force
deeplar train -o $outdir -m cnn --force