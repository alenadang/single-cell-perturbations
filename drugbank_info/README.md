External info extracted from drugbank, downloaded as an XML file
- the XML documentation can be found at this weblink: https://docs.drugbank.com/xml/#introduction

#

`makeDrugbankcsv.py` has functions defined to extract info based on the XML structure and constructs a pandas dataframe that is output and saved to your local folder as `drugbankConcise.csv` when run
- note that `drugbankFullDatabase.xml` is not uploaded to this repository

`exploreDrugbank.ipynb` gives a preview of the csv and shows the number of missing values for some of the key columns

#

This process was done to try and map the given compounds to their targets/pathways to augment the inputs to the trained model. *However as not all of the 144 compounds could be found in this database, since DrugBank only contains approved drugs, alternative methods were explored in the interest of time and efficiency.*

Potential use cases of this database could be to train a separate model to predict targets/pathways given SMILES sequences.

#

---supplementary info---

'144 compounds.csv' is the list of small molecules used in the experiment. 'SM_inDrugBank_checker.ipynb' is the code used to see whether compounds were able to be identified in the drugbank, and '144_compounds_checked.csv' is the csv file with the updated column which is a boolean indicating whether the small molecules are found in drugbank.


