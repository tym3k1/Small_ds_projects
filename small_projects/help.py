import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

dataset = pd.read_csv('default_card.csv', delimiter=';', header=1)

header = dataset.columns.values[1:]
corr = dataset[:12].corr()
print(corr)