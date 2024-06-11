# QuaKe Examples

## Pipelines

The commands in this section launch utility jobs from the QuaKe directory to reproduce a full pipeline.

In the following, it is assumed that the output folder (i.e. `../../output/tmp`)
already exists and can be created by `mkdir`.


- CNN + SVM training

    ```bash
    source examples/cnn_svm_pipeline.sh ../output/tmp
    ```

- Transformer + SVM training

    ```bash
    source examples/attention_svm_pipeline.sh ../output/tmp
    ```
- CNN + QSVM training

    ```bash
    source examples/cnn_qsvm_pipeline.sh ../output/tmp
    ```

- Transformer + QSVM training

    ```bash
    source examples/attention_qsvm_pipeline.sh ../output/tmp
    ```
## Comparing SVM and QSVM

The following command launches a training session for SVM and QSVM and produces customizable graphical outputs inside an external folder.

Default folder name : `../Training_results`.

From the QuaKe directory:

    ```bash
    python examples/svm_vs_qsvm.py
    ```

## Genetic featuremaps

The following command launches a quantum featuremap optimization through a genetic optimization, and save results on an external folder.

Default folder name: `../Output_genetic`.

From the QuaKe directory:

    ```bash
    python examples/genetic_featuremap_v3.py
    ```
`genetic_featuremap_v3` loads feature extracted with an autoencoder.
- Autoencoder training

    ```bash
    quake train -o ../output/tmp -m autoencoder
    ```

## Classification accuracy of Convolutional and Attention NN

Folder `performance_images` contains the models classification accuracies for different spatial resolutions achieved by resampling the dataset.

## Run Parallel QSVM on QPU example
Prepare a Python environment and activate it. (It should work with any python version > 3.8)

clone the QuaKe repository:
    ```bash
    git clone https://github.com/qismib/QuaKe.git
    ```

    ```bash
    cd QuaKe
    pip install -e .
    ```

inside QuaKe/examples/torino_run.py (right after the imports), fill the IBM account/provider fields with your token/instance as required

Move inside the examples folder and run the script. This will prepare the Quantum Circuits and send batch jobs to IBM Torino
    ```bash
    cd examples
    python torino_run.py
    ```