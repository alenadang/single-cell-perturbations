#!/usr/bin/env python
# coding: utf-8

# # model 5

# In[6]:


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
# <br>map sm_name to affinities_MPNN_CNN_BindingDB.csv

# In[7]:


de_train = pl.scan_parquet('de_train.parquet')
de_train_df = de_train.collect().to_pandas()

# test provided by kaggle --> upload predictions to kaggle to get the score
id_map = pd.read_csv('id_map.csv')

affinities = pd.read_csv('affinities_Transformer_CNN_BindingDB.csv', index_col=0)


# train dataset provided by kaggle
# - will be split into train/test/validation for internal testing before model is trained on the entire train and used to predict on the test in id_map

# In[8]:


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


# In[9]:


# one-hot encode cell_type
cell_type = de_train_df['cell_type'].to_numpy().reshape(-1, 1)
encoder = OneHotEncoder()
encoder.fit(cell_type)
# of type scipy.sparse._csr.csr_matrix
encoded_cell_type = encoder.transform(cell_type)

# map sm_name to affinities
sm_name = de_train_df['sm_name']

# has shape (614, 12766), of type numpy.ndarray
np_encoded_affinities = extractAffinities(sm_name, affinities)

# concatenate encoded_cell_type and np_encoded_affinities
# final shape (614, 12772)
encoded_features = np.hstack((encoded_cell_type.toarray(), np_encoded_affinities))

# wanted output
genes_lfc = de_train_df.drop(columns=['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control'])


# In[10]:


# repeat for kaggle test set
kaggle_cell_type = id_map['cell_type'].to_numpy().reshape(-1, 1)
encoded_kaggle_cell_type = encoder.transform(kaggle_cell_type)

kaggle_sm_name = id_map['sm_name']
encoded_kaggle_affinities = extractAffinities(kaggle_sm_name, affinities)

# final shape (255, 12772)
encoded_kaggle_features = np.hstack((encoded_kaggle_cell_type.toarray(), encoded_kaggle_affinities))


# In[11]:


# Split the data into 70% training, 15% validation, and 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(encoded_features, genes_lfc.values, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# used for final training before predicting on kaggle_test
full_features = encoded_features
full_labels = genes_lfc.values


# ## model training

# In[12]:


from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential


# In[ ]:


tf.random.set_seed(42)

model_2 = Sequential([
    Dense(3400, activation="tanh"),
    Dense(3000, activation="tanh"),
    Dense(2100, activation="tanh"),
    Dense(200, activation="tanh"),
    Dense(200, activation="tanh"),
    Dropout(0.2),
    Dense(18211, activation="linear")
])

model_2.compile(loss="mae", 
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                metrics=[custom_mean_rowwise_rmse])

history_2 = model_2.fit(X_train, y_train,
                       epochs=10,
                       validation_data=(X_val,y_val),
                       batch_size=32,
                       callbacks=[create_model_checkpoint("model_2", monitor="val_custom_mean_rowwise_rmse")])



calculate_mae_and_mrrmse(model=model_2, data=X_test, y_true=y_test)


# In[21]:


plot_training_history(history_2, metrics=["custom_mean_rowwise_rmse"])

kaggle_prediction = model_2.predict(encoded_kaggle_features, batch_size=1)


sample_submission = pd.read_csv('./sample_submission.csv')
sample_submission.iloc[:,1:] = kaggle_predictions

sample_submission.to_csv('submission_model_3.csv', index=False)

