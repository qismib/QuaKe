# QuaKe Examples

## Pipelines

The commands in this section launch utility jobs from the DeepLAr directory to reproduce a full pipeline.

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
    
## Classification accuracy of Convolutional and Attention NN

Folder `performance_images` contains the models classification accuracies for different spatial resolutions achieved by resampling the dataset.