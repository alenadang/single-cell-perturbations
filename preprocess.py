import polars as pl
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

de_train = pl.scan_parquet('./kaggledata/de_train.parquet')
de_train_df = de_train.collect().to_pandas()
