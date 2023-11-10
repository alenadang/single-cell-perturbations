# predict binding affinities for each compound and store in affinities.csv
# if default model to be changed, have to input drug_encoding, target_encoding, trained_dataset in command line
# default model is MPNN_CNN_BindingDB

from DeepPurpose import utils, dataset
from DeepPurpose import DTI as models
import pandas as pd
import numpy as np
import warnings
import sys
warnings.filterwarnings("ignore")

# model names consist of drug_encoding, target_encoding, trained_dataset
# ref https://github.com/kexinhuang12345/DeepPurpose/blob/master/README.md#pretrained-models for full model list
if len(sys.argv) == 4:
    drug_encoding = sys.argv[1]
    target_encoding = sys.argv[2]
    trained_dataset = sys.argv[3]

else: drug_encoding, target_encoding, trained_dataset = 'MPNN', 'CNN', 'BindingDB'

model_name = f'{drug_encoding}_{target_encoding}_{trained_dataset}'
model = models.model_pretrained(model = model_name)

print('Model used: ' + model_name)

# read in files
drugs = pd.read_csv('smiles.csv') #146 including 2 control compounds
peptides = pd.read_csv('peptides2.csv')

X_target = peptides['peptide'].tolist() #12766

# create empty dataframe to store affinities
num_rows = len(drugs)
col_names = ['sm_name', 'SMILES'] + peptides['location'].tolist()

affinities = pd.DataFrame(columns = col_names, index = range(num_rows))

print(f'Empty dataframe created with {num_rows} rows and {len(col_names)} columns')
print('Starting to predict affinities...')
for i in range(num_rows):
    print(f'Predicting affinities for compound {i+1} of {num_rows}')    
    smiles = drugs['SMILES'][i]
    affinities['SMILES'][i] = smiles
    affinities['sm_name'][i] = drugs['sm_name'][i]

    X_drug = np.repeat(smiles, len(X_target))   
    y = np.repeat(5000, len(X_target)) # y can be arbitrary, does not affect predicted value

    X_pred = utils.data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='no_split')

    y_pred = model.predict(X_pred)

    affinities.iloc[i, 2:] = y_pred

print('Finished predicting affinities')

filename = f'affinities_{model_name}.csv'
affinities.to_csv(filename)

print(f'Affinities for {model_name} exported to {filename}')