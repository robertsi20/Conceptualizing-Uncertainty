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
We construct two Datastreams for our experiments and Case Study:
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

# Step 3a: Experiments
To see the results of the experiments, one can either run the experiment notebooks (assuming one has acquired the data) or directly load the csv files which contain the results over 50 runs used in the paper. 
The scripts in the notebooks construct the streams and drifts using the datasets obtained above. 

We do note that there are more models here than presented in the paper. We ultimately chose "one_local_l_probs" as Model h \tilde, because it performed the best while maintaining a high degree of interpretability. 
The other models are produced by counting more concepts and could potentially be used in certain settings.

To see examples of the produced explanantions, please follow along in the case study notebook.

