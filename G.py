from imblearn.over_sampling import SMOTE
from DatasetsLoader import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.random import sample_without_replacement
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold import trustworthiness
import numpy as np
from plots.Tsne import CTGANTsne
import pandas as pd

obj = CTGANTsne(dataset_name="CreditCard")
obj.fit()

num_samples = obj.X_real.shape[0]
synthetic_data_x = obj.CTGAN.sample(num_samples)
synthetic_data_y = np.ones(num_samples)
X_train_ctgan = synthetic_data_x

if obj.X_real.shape[0] <= 75:
    perplexity = obj.X_real.shape[0] / 2
else:
    perplexity = 50

X_total = np.concatenate((obj.X_real, X_train_ctgan), axis=0)
df = pd.DataFrame(X_total)
df.to_csv("e.csv", header=False, index=False)


