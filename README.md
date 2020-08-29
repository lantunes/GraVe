# GraVe

This code in this project can be used as a starting point for creating node-level graph embeddings using the GloVe 
approach. Much work in this space appears to have been done using the word2vec approach, with either the SkipGram or 
CBOW models, but very little has been done with the GloVe approach. 

This project makes use of the original [DeepWalk](https://github.com/phanein/deepwalk) implementation to generate the
graph random walk corpus, and the [glove_python](https://github.com/maciejkula/glove-python) project to learn the 
GloVe embeddings from the random walks. 

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