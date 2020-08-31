# GraVe

This code in this project can be used as a starting point for creating node-level graph embeddings using the GloVe 
approach. Much work in this space appears to have been done using the word2vec approach, with either the SkipGram or 
CBOW models, but very little has been done with the GloVe approach. 

This project makes use of the original [DeepWalk](https://github.com/phanein/deepwalk) implementation to generate the
graph random walk corpus, and the [glove_python](https://github.com/maciejkula/glove-python) project to learn the 
GloVe embeddings from the random walks.

![Results](resources/grave_results.png)

Generating walks, example usage:
```
python walks.py --input examples/karate.adjlist --output karate
```
This generates a file, `karate.walks.0`, which contains the corpus.

Creating the GloVe embeddings, example usage:
```
python glove_training.py --create karate.walks.0 --components 2  --train 10
```
This generates two files: `corpus.model` and `glove.model`.

Inspecting a trained GloVe model, example usage:
```
python inspect_model.py --model glove.model --adjlist examples/karate.adjlist
```

The plots below are color-coded according to the node degree: 

![Results by node degree](resources/grave_results_by_degree.png)

The plots below are color-coded according to community structure:

![Results by community](resources/grave_results_by_community.png)

The embeddings for the first plot were created with a symmetric context window size of 10, whereas the embeddings for 
the second plot were created with a symmetric context window size of 1.

The plots below are color-coded according to community structure, and the embeddings were generated from a walk dataset 
that was created with the [node2vec](https://github.com/aditya-grover/node2vec) approach:

![Results node2vec](resources/grave_results_node2vec.png)

The embeddings were created with a symmetric context window size of 10.

