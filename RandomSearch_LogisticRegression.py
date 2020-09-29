import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
diabetes = pd.read_csv("C:/Users/anuno/OneDrive/Documents/ITESO/PAP 2/Data/diabetes.csv",header=None,names=['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label'])
diabetes =diabetes[1:]
x=diabetes[['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']]
y=diabetes[["label"]]
logreg = LogisticRegression()
logreg.fit(x,y)
