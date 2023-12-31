Some models were tuned using Keras Tuner with the defined search space indicated in bullets below the respective models

Files required for tuning and training are in `Models/Files_Required`. Include files in same directory when retraining models. Due to the stochastic tuning of Keras Tuner, different hyperparameters may be achieved with each tuning of the model.

When jupyter notebook and python files are present with the same names, python files take precedence.

#

`compareAffinities.ipynb` was used to see if there were any differences in accuracy of models when different encoders (from DeepPurpose) were used to predict the protein binding affinities. Architecture from Model 5, was used to train the models.


#

Model 0: One-Hot Encoded Compounds

Model 1: Incoporation of Protein Affinities

Model 2: Experimented with Keras Tuner. Model not saved and cannot be opened.

Model 3: Tuned with Keras Tuner. Retrained with loss function Mean Absolute Error.
- 1-5 layers
- 32-18211 nodes
- relu/tanh activation function
  
Model 4: Tuned with Keras Tuner. Retrained with loss function Mean Absolute Error.
- 3-5 layers
- 5000-18211 nodes
- relu/tanh/linear activation functions
- optional dropout layer (of dropout rate 0.25)

Model 5: Tuned with Keras Tuner. Retrained with loss function Mean Absolute Error.
- 3-5 layers
- 200-10000 nodes
- relu/tanh/linear activation functions
- optional dropout layer (of dropout rate 0.25)

Model 6: Tuned with Keras Tuner with loss function Mean Absolute Error.
- 3-5 layers
- 200-8000 nodes
- relu/tanh/linear activation functions
- optional dropout layer (of dropout rate 0.25)
