# Signed Graph Augmentation (SGA)

>In this project, we propose a graph augmentaiton method for Signed Graph Neural Networks  
![image](https://github.com/Alex-Zeyu/SGA/blob/main/framework.png)

## Abstract
Signed Graph Neural Networks (SGNNs) play a crucial role in the analysis of intricate patterns within real-world signed graphs, where both positive and negative links coexist. Nevertheless, there are three critical challenges in current signed graph representation learning using SGNNs. First, signed graphs exhibit significant sparsity, leaving numerous latent structures uncovered. Second, SGNN models encounter difficulties in deriving proper representations from unbalanced triangles. Finally, real-world signed graph datasets often lack supplementary information, such as node labels and node features. These challenges collectively constrain the representation learning potential of SGNN. We aim to address these issues through data augmentation techniques. However, the majority of graph data augmentation methods are designed for unsigned graphs, making them unsuitable for direct application to signed graphs. To the best of our knowledge, there are currently no data augmentation methods specifically tailored for signed graphs. In this paper, we propose a novel <ins>S</ins>igned <ins>G</ins>raph <ins>A</ins>ugmentation framework, **SGA**. This framework primarily consists of three components. In the first part, we utilize the SGNN model to encode the signed graph, extracting latent structural information in the encoding space, which is then used as candidate augmentation structures. In the second part, we analyze these candidate samples (i.e., edges), selecting the most beneficial candidate edges to modify the original training set. In the third part, we introduce a new augmentation perspective, which assigns training samples different training difficulty, thus enabling the design of new training strategy. Extensive experiments on six real-world datasets, i.e., Bitcoin-alpha, Bitcoin-otc, Epinions, Slashdot, Wiki-elec and Wiki-RfA show that SGA improve the performance on multiple benchmarks. Our method outperforms baselines by up to 22.2% in terms of AUC for SGCN on Wiki-RfA, 33.3% in terms of F1-binary, 48.8% in terms of F1-micro, and 36.3% in terms of F1-macro for GAT on Bitcoin-alpha in link sign prediction.

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
python dataAugmentation.py --pos_del 0.4 --neg_del 0.45 --pos_add 0.93 --neg_add 0.93
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
