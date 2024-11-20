# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
# a6q1

import pandas as pd
import matplotlib.pyplot as plt 
from statsmodels.graphics.tsaplots import plot_acf
import warnings

# reading csv
warnings.filterwarnings('ignore')
df = pd.read_csv('daily_covid_cases.csv')

# creating line plot
df.plot.line('Date','new_cases')
plt.xlabel('Dates')
plt.ylabel('Number of Covid-19 cases')
plt.show()

#shifting by one day
l1 = df.shift(1)
print('Correlation between the given time sequence and one-day lagged generated sequence is',round(df['new_cases'].corr(l1['new_cases']),3))

# scatter plot
plt.scatter(df['new_cases'], l1['new_cases'])
plt.show()

# Pearson correlation coefficient calculation and plotting
c=[]
for l in range(1,7):
    lag_df=df.shift(l)
    cor=df['new_cases'].corr(lag_df['new_cases'])
    print('The Pearson correlation coefficient for',l,'days lag:',round(cor,3))
    c.append(cor)
plt.plot(range(1,7),c)
plt.show()

# correlogram plot
plot_acf(df['new_cases'])
plt.show()
