# script to convert Ensembl IDs to peptide sequences

library(tidyr)
library(dplyr)
library(biomaRt)

# columns 1:5 --> meta data
de_train <- read.csv('de_train_df.csv')
gene_info <- read.csv('multiome_var_meta.csv')

# 18211 unique genes
gene_names <- colnames(de_train)[6:length(colnames(de_train))]

# round 1 filtering (15011)
v1 <- gene_info %>% filter(location %in% gene_names)
missingv1 <- setdiff(gene_names, v1$location)

# change '.' to '-'
missingv1 <- gsub('\\.', '-', missingv1)

# round 2 filtering (568)
v2 <- gene_info %>% filter(location %in% missingv1)
missingv2 <- setdiff(missingv1, v2$location) # 2632 genes unmapped

# 15579 mapped genes
total_mapped <- rbind(v1, v2)

mart <- useEnsembl('ensembl', dataset="hsapiens_gene_ensembl")

# map in batches of 500 to prevent the time out error
# except last 579 done together
m1 <- getSequence(id=total_mapped$gene_id[15001:15579], type='ensembl_gene_id', 
                  seqType='peptide', mart=mart)

batches <- seq(1, 15500, 500)

m2 <- m1

for (i in 1:(length(batches)-1)) {
  print(paste0('Retrieving batch ', i))
  current_batch <- getSequence(id=total_mapped$gene_id[batches[i]:(batches[i+1]-1)], 
                               type='ensembl_gene_id', seqType='peptide', mart=mart)
  m2 <- rbind(m2, current_batch)
}

# 12786 unique gene ids
# 72737 unique peptides
sequences <- filter(m2, peptide != 'Sequence unavailable') %>% 
  distinct(peptide, .keep_all = TRUE) # only keep first instance of peptide

result_df <- sequences %>%
  left_join(gene_info %>% 
              select(gene_id, location), by = c("ensembl_gene_id" = "gene_id"))

# sort by the gene name (in location column) and export
result_df <- result_df[order(result_df$location),]
write.csv(result_df, 'peptides.csv')

# keep only first instance of each ensembl ID
# 12766 unique peptides left
distinct_genes <- result_df %>% distinct(ensembl_gene_id, .keep_all = TRUE)
write.csv(distinct_genes, 'peptides2.csv')