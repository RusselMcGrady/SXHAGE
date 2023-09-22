# This is the repo. for the conference paper of JMS: Self-X Heterogeneous Attributed Graph Embedding-based Configuration Framework for Cognitive Mass Personalization  

# How to code an SXHAGE model in PyTorch
## PyTorch implementation of the code for multi-criteria attributed graph embedding. The self-attention-based consistent layer is implemented.

This is the repo of the MCAGE model for link prediction on heterogeneous KG

The train_SXHGAT.py file shows the graph clustering task on the data from the .csv file in "/data".

The train_SXHGAT_LP.py file contains the link prediction task. 


## Notice: you must revise the 241th line of the RandomLinkSplit to declare if the graph is directed or undirected

## install the required package in requirements.txt
pip install -r requirements.txt

## train
python train_SXHGAT.py
