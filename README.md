# Do LLMs "know" internally when they follow instructions?

This software project accompanies the research paper 'Do LLMs "know" internally when they follow instructions?': https://openreview.net/forum?id=qIN5VDdEOr

The software includes code to model whether an LLM followed instructions by training probes on the internal weights of the model, and code to engineer representations based on the ‘instruction-following’ direction identified by the probe to improve instruction-following performance.  The repo also includes a dataset (modified from IFEval, a common instruction following dataset) designed specifically to disentangle to separate instruction-following ability from task execution.

LICENSE specifies the license for the included code, and LICENSE_DATASET for the included dataset.

Please cite this work as:

Heo, J., Heinze-Deml, C., Elachqar, O., Ren, S. Y., Chan, K. H. R., Nallasamy, U., Miller A., & Narain, J. Do LLMs internally "know" when they follow instructions?. In _ICLR 2025._

## 1. Save Activations of LLMs

To save activations from the LLMs, run the following script:

```bash
bash script/save_LLMs_activations.sh
```

## 2. Evaluating Responses in Instruction Following

For instructions on evaluating LLM responses in instruction-following tasks with IFEval, please refer to the IFEval (Instruction Following Evaluation) repository for detailed guidelines and methodology. https://huggingface.co/datasets/google/IFEval

## 3. Training a Linear Probe

To train a linear probe on the saved activations, use the provided notebook: notebooks/train_linear_prob.ipynb


## 4. Run Representation Engineering

To run representation engineering and analyze learned model representations, use the following command:
```bash
bash script/representation_engineering.sh
```

## Datasets

The dataset IFEval-Simple (modified from IFEVal) referenced in the paper is in the 'data' subfolder in .json format
