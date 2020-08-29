# GraVe

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