#!/usr/bin/env python
# coding: utf-8


# In[1]:


import polars as pl
import numpy as np
import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# import from the auxFunctions.py file
from auxFunctions import calculate_mae_and_mrrmse, mean_rowwise_rmse_loss, custom_mean_rowwise_rmse, create_model_checkpoint, plot_training_history


# ## pre-process data
# one-hot encode cell type
# <br>map sm_name to affinities_Transformer_CNN_BindingDB.csv

# In[2]:


de_train = pl.scan_parquet('./de_train.parquet')
de_train_df = de_train.collect().to_pandas()

# test provided by kaggle --> upload predictions to kaggle to get the score
id_map = pd.read_csv('./id_map.csv')

affinities = pd.read_csv('affinities_Transformer_CNN_BindingDB.csv', index_col=0)


# train dataset provided by kaggle
# - will be split into train/test/validation for internal testing before model is trained on the entire train and used to predict on the test in id_map

# In[3]:


def extractAffinities(sm_names, affinities):
    """
    Function to extract affinities from the affinities dataframe

    Parameters:
    - sm_names: List/Array of sm_names
    - affinities: Stored affinities predicted using DeepPurpose

    Returns:
    - Affinities as a numpy array
    """
    encoded_affinities = []
    for name in sm_names:
        filtered = affinities[affinities['sm_name'] == name]
        sm_affinities = filtered.iloc[:, 2:].values[0]
        encoded_affinities.append(sm_affinities)

    np_encoded_affinities = np.array(encoded_affinities)

    return np_encoded_affinities


# In[4]:


# one-hot encode cell_type
cell_type = de_train_df['cell_type'].to_numpy().reshape(-1, 1)
encoder = OneHotEncoder()
encoder.fit(cell_type)


# of type scipy.sparse._csr.csr_matrix
encoded_cell_type = encoder.transform(cell_type)

# map sm_name to affinities
sm_name = de_train_df['sm_name']

# has shape (614, 12766), of type numpy.ndarray
#this is the training set
np_encoded_affinities = extractAffinities(sm_name, affinities)

# concatenate encoded_cell_type and np_encoded_affinities
# final shape (614, 12772)
encoded_features = np.hstack((encoded_cell_type.toarray(), np_encoded_affinities))

print(encoded_cell_type)


# wanted output
genes_lfc = de_train_df.drop(columns=['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control'])


# In[5]:


# repeat for kaggle test set
kaggle_cell_type = id_map['cell_type'].to_numpy().reshape(-1, 1)


encoded_kaggle_cell_type = encoder.transform(kaggle_cell_type)



kaggle_sm_name = id_map['sm_name']

#This is for the test data set
encoded_kaggle_affinities = extractAffinities(kaggle_sm_name, affinities)


# final shape (255, 12772)
encoded_kaggle_features = np.hstack((encoded_kaggle_cell_type.toarray(), encoded_kaggle_affinities))


# In[6]:


# Split the data into 70% training, 15% validation, and 15% testing
x_train, x_temp, y_train, y_temp = train_test_split(encoded_features, genes_lfc.values, test_size=0.3, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, shuffle=False)

# used for final training before predicting on kaggle_test
full_features = encoded_features
full_labels = genes_lfc.values

print(y_train)


# ## model tuning

# In[7]:


from tensorflow import keras
from tensorflow.keras import layers

from keras import Sequential
from keras.layers import Activation, Dense


# In[8]:


##define search space

def build_model(hp):
    model = keras.Sequential()

    for i in range(hp.Int("num_layers", 3, 5)):
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=5000, max_value=18211, step=1000),
                activation=hp.Choice("activation", ["relu", "tanh", "linear"]),
            )
        )

    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    
    model.add(layers.Dense(18211, activation="linear"))


    model.compile(
        optimizer="adam", loss=mean_rowwise_rmse_loss, metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    return model


# In[9]:


import keras_tuner


# In[10]:


build_model(keras_tuner.HyperParameters())


# In[11]:


#select tuner classs to run search

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective=keras_tuner.Objective("val_root_mean_squared_error", direction="min"),
    max_trials=15,
    executions_per_trial=2,
    overwrite=False,
    directory="models",
    project_name="model_4",
)


# In[12]:


#search space summary

tuner.search_space_summary()


# In[ ]:


tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))

best_hps = tuner.get_best_hyperparameters(5)



# In[ ]:


models = tuner.get_best_models(num_models=2)
best_model = models[0]
# Build the model.
# Needed for `Sequential` without specified `input_shape`.
best_model.build(input_shape=(614, 12772))
best_model.summary()


# In[ ]:


best_hps = tuner.get_best_hyperparameters(5)
# Build the model with the best hp.
model_tuned = build_model(best_hps[0])
# Fit with the entire dataset.
x_all = np.concatenate((full_features))
y_all = np.concatenate((full_labels))
model_tuned.fit(x=x_all, y=y_all, epochs=30)


# In[ ]:


model_tuned.save("model_4")
model_tuned.save_weights("model_4_weights.h5")

calculate_mae_and_mrrmse(model=model_tuned, data=x_test, y_true=y_test)







