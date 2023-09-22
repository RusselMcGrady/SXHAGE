# This is the repo. for the conference paper of IEEE CASE 2023: Multi-criteria attributed graph embedding-enabled decision support system for resilient manufacturing (accepted)

# How to code an AGE model considering multi-criteria and multi-structure on heterogeneous KG in PyTorch
## PyTorch implementation of the code for multi-criteria attributed graph embedding. The self-attention-based consistent layer is implemented.

This is the repo of the MCAGE model for link prediction on heterogeneous KG

The train.py file shows how to use the model conduct link prediction on the data from the .csv file in "/data".

The test.py file contains the eval function. 


## Notice: you must revise the 241 line of the RandomLinkSplit to declare if the graph is directed or undirected

## install the required package in requirements.txt
pip install -r requirements.txt

## train
python train.py

## test
python test.py

