import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
housing_data=fetch_california_housing(as_frame=True)
df=housing_data.frame
print(df.head())
print("basic information about dataset")
print(df.info())
print("summary statics")
print(df.describe)
print("missing value in each colum")
print(df.isnull().sum())
variable_meaning={
    "Medinc":"median income of block group",
    "houseage":"avreage houseage of block group",
    "averooms":"average no. of rooms per household",
    "avebedrms":"average no. of bedrooms per household",
    "population":"population of block group",
    "aveoccup":"averge of household member",
    "latitide":"latitide of b;ock group",
    "longitude":"longitude of block groip",
    "target":"the house value(in $100.000)"
}
variable_data=pd.DataFrame(list(variable_meaning.items()),columns=["feature","description"])
print("variable meaning table")
print(variable_data)
correlation_matrix=df.corr()
print("correlational matrix")
print(correlation_matrix)
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt='.2f',linewidth=0.5)
plt.show()
sns.pairplot(df,diag_kind='kde',plot_kws={'alpha':0.7})
print("key insights")
print("dataset ha",df.shape[0],"colums",df.shape[1])
