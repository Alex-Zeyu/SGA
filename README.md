# Signed Graph Augmentation (SGA)

>In this project, we propose a graph augmentaiton method for Signed Graph Neural Networks

Experimental steps:
1. Train an SGCN model using the original training set to generate node embeddings.
2. Use the node embeddings to generate candidate lists for deleting positive edges, deleting negative edges, adding positive edges, and adding negative edges.
3. Set probability threshold parameters and implement data augmentation on the training set data based on the candidate lists.
4. Calculate the difficulty scores of all edges of the training set after data augmentation, and set up a training plan.
5. Use the training set after data augmentation to train baselines according to the training plan.
