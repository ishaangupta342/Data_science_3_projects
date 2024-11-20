# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1=pd.read_csv("landslide_data3_miss.csv")

y = list(df1.isnull().sum())
x = list(df1)
fig=plt.figure(figsize=(10,5))
plt.bar(x,y,color='orange',width=0.5)
plt.xlabel("attributes")
plt.ylabel("No. of missung values")
plt.title("bar graph depicting no. of missing values of attributes")
plt.show()

