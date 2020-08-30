The `karate.*` walk and model files in this directory were generated with the following hyperparameters:
- number of walks: 10
- walk length: 40
- GloVe:
  - context window size: 10
  - embedding components: 2
  - learning rate: 0.05
  - training epochs: 10
  - workers: 1
  - alpha: 0.75
  - max_count: 100
  - max_loss: 10.0
  
The `karate2.*` walk and model files in this directory were generated with the following hyperparameters:
- number of walks: 160
- walk length: 40
- GloVe:
  - context window size: 1
  - embedding components: 2
  - learning rate: 0.05
  - training epochs: 200
  - workers: 1
  - alpha: 0.75
  - max_count: 100
  - max_loss: 10.0

Compared to the `karate` model, the `karate2` model was trained on a larger walk dataset, as well as with many more 
epochs and a symmetric context window size of 1. The embeddings in the `karate` model appear to be clustered according 
to node degree, whereas the embeddings in the `karate2` model appear to be clustered according to community structure.   