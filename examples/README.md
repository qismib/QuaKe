# QuaKe Examples

## Pipelines

The commands in this section launch utility jobs to reproduce a full pipeline.

In the following, it is assumed that the output folder (i.e. `../../output/tmp`)
already exists and can be created by `mkdir`.

- CNN + SVM training

    ```bash
    ./cnn_svm_pipeline.sh ../../output/tmp
    ```

- Transformer + SVM training

    ```bash
    ./attention_svm_pipeline.sh ../../output/tmp
    ```
