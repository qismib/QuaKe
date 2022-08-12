# QuaKe Examples

## Pipelines

The commands in this section launch utility jobs from the QuaKe directory to reproduce a full pipeline.

In the following, it is assumed that the output folder (i.e. `../../output/tmp`)
already exists and can be created by `mkdir`.


- CNN + SVM training

    ```bash
    examples/cnn_svm_pipeline.sh ../output/tmp
    ```

- Transformer + SVM training

    ```bash
    examples/attention_svm_pipeline.sh ../output/tmp
    ```
- CNN + QSVM training

    ```bash
    examples/cnn_qsvm_pipeline.sh ../output/tmp
    ```

- Transformer + QSVM training

    ```bash
    examples/attention_qsvm_pipeline.sh ../output/tmp
    ```
## Comparing SVM and QSVM

The following command launches a training session for SVM and QSVM and produces customizable graphical outputs inside an external folder.
Default folder: `../Training_results`.

    ```bash
    python examples/svm_vs_qsvm.py
    ```

## Genetic generative featuremaps

The following command launches a quantum featuremap optimization through a genetic optimization, and save results on an external folder.
Default folder: `../genetic_featuremap`.

    ```bash
    python examples/genetic_featuremap.py
    ```

## Classification accuracy of Convolutional and Attention NN

Folder `performance_images` contains the models classification accuracies for different spatial resolutions achieved by resampling the dataset.