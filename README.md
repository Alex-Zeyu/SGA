# Signed Graph Augmentation (SGA)

>In this project, we propose a graph augmentaiton method for Signed Graph Neural Networks

Experimental steps:
1. Train an SGCN model using the original training set.
2. Use the trained SGCN model to generate candidate lists for deleting positive edges, deleting negative edges, adding positive edges, and adding negative edges.
3. Set probability threshold parameters and implement data augmentation on the training set data based on the candidate lists.
4. Calculate the difficulty scores of all edges of the training set after data augmentation, and setup a training plan.
5. Use the training set after data augmentation to train baselines according to the training plan.
