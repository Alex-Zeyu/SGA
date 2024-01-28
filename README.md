# Signed Graph Augmentation (SGA)

>In this project, we propose a graph augmentaiton method for Signed Graph Neural Networks

![image](https://github.com/Alex-Zeyu/SGA/blob/main/framework.png)

## Abstract
Signed Graph Neural Networks (SGNNs) are essential for analyzing complex patterns in real-world signed graphs, where positive and negative links coexist. Although significant progress has been made in the field of SGNNs research, two issues persist in the current SGNN-based signed graph representation learning models. First, real-world datasets exhibit significant sparsity, leaving many latent structures unexplored. Second, the majority of SGNN models, which are based on balance theory, cannot learn proper representations from unbalanced triangles commonly found in real-world signed graph datasets. We aim to address the issues through data augmentation techniques. Current graph augmentation methods are primarily designed for unsigned graphs and cannot be directly applied to signed graphs. To the best of our knowledge, there are currently no augmentation methods specifically tailored for signed graphs. In this paper, we propose a novel <ins>S</ins>igned <ins>G</ins>raph <ins>A</ins>ugmentation method, **SGA**. The method consists of three steps. Firstly, obtain node embeddings through SGNN model, then extract potential candidate edges in the encoding space. Subsequently, analyze these candidate edges and choose the beneficial ones to modify the original graph. At last, SGA introduces a new augmentation perspective, which assigns training samples different training difficulty, thus enabling the design of new training strategy. Extensive experiments on six real-world datasets, i.e., Bitcoin-alpha, Bitcoin-otc, Epinions, Slashdot, Wiki-elec and Wiki-RfA show that SGA improve the performance on multiple benchmarks. Our method outperforms baselines by up to 14.8\% in terms of AUC for SGCN on Wiki-RfA, 26.2\% in terms of F1-binary, 32.3\% in terms of F1-micro, and 24.7\% in terms of F1-macro for SGCN on Slashdot in link sign prediction.

## Requirements
```
numpy==1.24.2
scipy==1.11.2
pandas==2.1.0
torch==1.12.0
torch_geometric==2.3.1
torch-geometric-signed-directed==0.22.1
tqdm==4.65.0
```

## Experimental steps
1. Train an SGCN model using the original training set to generate node embeddings.
```
python embeddingsGenerator.py --dataset wiki-RfA --num 1
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;You can use different datasets and different training set splits.  

2. Use the node embeddings to generate candidate lists for deleting positive edges, deleting negative edges, adding positive edges, and adding negative edges.
```
python candidatesGenerator.py --dataset wiki-RfA --num 1
```
3. Set probability threshold parameters and implement data augmentation on the training set data based on the candidate lists.
```
python dataAugmentation.py --pos_del 0.4 --neg_del 0.45 --pos_add 0.98 --neg_add 0.98
```
4. Calculate the difficulty scores of all edges of the training set after data augmentation, and set up a training plan.Use the training set after data augmentation to train baselines according to the training plan.
```
python sgcn_SGA.py --T 30 --lambda0 0.25
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;You can only implement the data augmentation part of SGA by
```
python sgcn_SGA.py --useDataAugmentation 1 --useTrainingPlan 0
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;or only implement the training plan part of SGA by
```
python sgcn_SGA.py --useDataAugmentation 0 --useTrainingPlan 1
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;or just implement the baseline by
```
python sgcn_SGA.py --useDataAugmentation 0 --useTrainingPlan 0
```
