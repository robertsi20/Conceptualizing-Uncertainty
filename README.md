# Conceptualizing-Uncertainty

This repository contains code for the paper:  
**Conceptualizing Uncertainty, Isaac Roberts, Alexander Schulz, Sarah Schroeder, Fabian Hinder, Barbara Hammer, submitted to the International Conference on Neural Information Processing 2025
**

# Step 1: Technical Pre-Requisites

This experiment builds mainly on the following code:
- [CRAFT in Xplique]((https://github.com/deel-ai/xplique)) - A framework for automatically extracting Concept Activation Vectors which explain deep
  neural networks. Link to Paper [CRAFT](https://arxiv.org/abs/2211.10154)
- Uncertainty functions can be found from [https://git.cs.uni-paderborn.de/mhshaker/ida_paper114]. Link to [paper](https://arxiv.org/abs/2001.00893)

In this experiment, we use CRAFT in combination with [Pytorch](https://pytorch.org/).We suggest to setup a local [Conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)-environment
using **Python 3.10** and install the repository as follows:
```bash
git clone https://github.com/robertsi20/ConceptualizingConceptDrift.git
pip install datasets
pip install jupyter
pip install xplique
pip install timm
pip install fastai
```

For experiment 3 it is necessary to install **transformers** clone the [Cockatiel repo](https://github.com/fanny-jourdan/cockatiel) (we assume the cockatiel repo is located in the same directory as ours),
```bash
pip install transformers
git clone https://github.com/fanny-jourdan/cockatiel.git
```

and optionally to save images of the token attribution install imgkit and [wkhtmltopdf](https://wkhtmltopdf.org/).
```bash
pip install imgkit
```

# Step 2: Data Acquisition
- [NINCO](https://github.com/j-cb/NINCO) - The NINCO (No ImageNet Class Objects) dataset consists of 64 OOD classes with a total of 5879 samples. The OOD classes were selected to have no categorical overlap with any classes of ImageNet-1K.  Link to Paper [NINCO]((https://arxiv.org/abs/2306.00826))
- [Subset of ImageNet](https://www.image-net.org/) - ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images.  Link to Paper [ImageNet](https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf)
    - We focus our work on the ImageWoof and Imagenette subsets given by [fastai](https://github.com/fastai/imagenette)
        

To proceed with the NINCO datset:
```bash
mkdir data/
wget -O data/ninco.tar.gz https://zenodo.org/record/8013288/files/NINCO_all.tar.gz?download=1
mkdir data/ninco_data
tar -xf data/ninco.tar.gz -C data/ninco_data
```

# Step 3: Experiments
To see the results of the experiments, one can either run the experiment notebooks (assuming one has acquired the data) or directly load the csv files which contain the results. 

## Distinguishing Sources of Uncertainty Experiment
One can either run the Distinguish_Unc_Experiment.ipynb or look into the Distinguish_Unc_Experiment_results folder for all of the results which are included in the submitted paper or the Arxiv version. Link to Arxiv (([Paper](https://arxiv.org/abs/2503.03443)))

## Rejection Experiment
One can either run the Rejection_Experiment.ipynb or look into the Rejection_Experiment_results folder for all of the results which are included in the submitted paper or the Arxiv version. Link to Arxiv (([Paper](https://arxiv.org/abs/2503.03443)))

## Fairness in LLMs Experiment
One can either run the Fairness_LLM_Experiment.ipynb or look into the Fairness_LLM_results folder for all of the results which are included in the submitted paper or the Arxiv version. Link to Arxiv (([Paper](https://arxiv.org/abs/2503.03443)))



