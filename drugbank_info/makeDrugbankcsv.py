import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

# get SMILES sequence given a DRUG element
# returns None if no SMILES sequence is found
def getsmiles(element):
    properties = element.find('{http://www.drugbank.ca}calculated-properties')
    smiles_sequence = np.nan
    for entry in properties:
        kind_tag = entry.find('{http://www.drugbank.ca}kind')
        if kind_tag is not None and kind_tag.text == 'SMILES':
            smiles_sequence = entry.find('{http://www.drugbank.ca}value').text
            break  # break once you find the SMILES sequence
    return smiles_sequence

# returns a list of all categories drug falls under
def getCategories(element):
    categories = element.find('{http://www.drugbank.ca}categories')
    categories_list = []
    for category in categories:
        categories_list.append(category.find('{http://www.drugbank.ca}category').text)
    return categories_list

# searching for tags nested under 'classification'
def getClassifications(element, cat):
    tag = "{http://www.drugbank.ca}classification/{http://www.drugbank.ca}" + cat
    found = element.find(tag)
    if found is None: 
        return np.nan
    return found.text

# searching for target id and name
def getTarget(element):
    target = element.find('{http://www.drugbank.ca}targets/{http://www.drugbank.ca}target')
    if target is None:
        return np.nan, np.nan
    id = target.find('{http://www.drugbank.ca}id').text
    name = target.find('{http://www.drugbank.ca}name').text
    return id, name
    
# parsing XML file    
tree = ET.parse('drugbankFullDatabase.xml')
root = tree.getroot()

# extracting only small molecules which have SMILES sequences
# 12,227 small molecules as of 2023-01-04 when database was exported
small_molecules = []
current_index = 0

for child in root:
    if (child.attrib['type'] == 'small molecule'):
        small_molecules.append(current_index)
    current_index += 1

num_rows = len(small_molecules)
col_names = ['drugbank_id', 'name', 'SMILES', 'direct_parent', 'kingdom', 
             'superclass', 'class', 'subclass', 'target_id', 'target_name', 'categories']

drugbank_df = pd.DataFrame(columns = col_names, index = range(num_rows))
                
for i in range(0, len(small_molecules)):
    drug = root[small_molecules[i]]
    drugbank_df['drugbank_id'][i] = drug.find('{http://www.drugbank.ca}drugbank-id').text
    drugbank_df['name'][i] = drug.find('{http://www.drugbank.ca}name').text
    drugbank_df['SMILES'][i] = getsmiles(drug) # 644 sm molecules w/o SMILES sequences
    drugbank_df['direct_parent'][i] = getClassifications(drug, 'direct-parent')
    drugbank_df['kingdom'][i] = getClassifications(drug, 'kingdom')
    drugbank_df['superclass'][i] = getClassifications(drug, 'superclass')
    drugbank_df['class'][i] = getClassifications(drug, 'class') # 2646 sm molecules w/o drug classes
    drugbank_df['subclass'][i] = getClassifications(drug, 'subclass')
    drugbank_df['target_id'][i], drugbank_df['target_name'][i] = getTarget(drug) # 4862 sm molecules w/o targets
    drugbank_df['categories'][i] = getCategories(drug) # 4861 sm molecules w/o categories

drugbank_df.to_csv('drugbankConcise.csv', index = False)