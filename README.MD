# Grocery Product Classification

University assignment for the **Image Processing and Computer Vision** course. The project explores image classification on the [GroceryStoreDataset](https://github.com/marcusklasson/GroceryStoreDataset), which contains smartphone photographs of grocery products divided into **43 classes**.

> **Academic project only**
>
> This repository is not a ready-to-use application or reusable Python package. It does not provide a CLI, an inference script, a trained-model download, an API, or a deployment pipeline. The code is collected in a notebook and was created to document the experiments required for a university exam.

## What the notebook does

The project is implemented in [`assignment_2.ipynb`](assignment_2.ipynb) and contains two main experiments:

1. **Custom CNN from scratch**
   - image preprocessing and data augmentation;
   - a custom architecture based on convolutional and Inception-style blocks;
   - training with early stopping and learning-rate scheduling;
   - an ablation study covering depth, dropout, stem design, bottlenecks and data augmentation.

2. **ResNet-18 transfer learning**
   - replacement of the original classification head;
   - comparison between head-only training and full fine-tuning;
   - separate learning rates for the feature extractor and classifier.

The notebook also measures model size and computational cost in parameters, MACs and FLOPs.

## Results

The best validation accuracies recorded in the included experiment files are approximately:

| Model | Best validation accuracy |
|---|---:|
| Custom CNN, 3-stage variant | 70.27% |
| Fine-tuned ResNet-18 with custom head | 91.55% |

These values refer to the specific experimental setup stored in the notebook and are not intended as a reproducible benchmark.

## Repository contents

- `assignment_2.ipynb` — complete implementation and discussion;
- `abl_results.csv` and `abl_histories.csv` — ablation-study results and histories;
- `resnets_results.csv` and `resnet_histories.csv` — ResNet experiment results and histories;
- `alb_history.png` and `resnet_history.png` — training-history plots;
- `alb_train_logs.txt` and `resnet_train_logs.txt` — saved training logs.

## Running the notebook

The simplest way to inspect the project is through Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanieleBaiocco/IPCV_assignment2/blob/main/assignment_2.ipynb)

The first cells clone the dataset and install some notebook dependencies. Run the notebook from top to bottom in a Python environment with PyTorch and torchvision available. Training all configurations can require considerable time and benefits from a CUDA-compatible GPU.

## Limitations

- trained weights are not included;
- there is no command-line interface or standalone prediction program;
- experiments are tightly coupled to the notebook and Google Colab workflow;
- dependency versions are not pinned;
- the repository is intended for study and review, not production use.
