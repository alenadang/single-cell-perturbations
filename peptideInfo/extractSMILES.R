# script to extract SMILES sequences from de_train

library(tidyr)
library(dplyr)

de_train <- read.csv('de_train_df.csv')

smiles <- de_train %>% select(sm_name, sm_lincs_id, SMILES, control) %>% distinct(SMILES, .keep_all = TRUE)

write.csv(smiles, 'smiles.csv')