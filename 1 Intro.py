import pandas as pd #manipulate data in grid format (like excel)
import seaborn as sns #visualize data in plots
import numpy as np #broadcast logic to numpy array for faster runtime
import matplotlib.pyplot as plt

#matplotlib inline

#pandas
df = pd.read_csv(r'C:\Users\anton\Desktop\tut\zillow.csv') #df means dataframe
print(df.head(2))   #print top two rows of the file
df.shape #2d size of the df (15000,19). no () because it's an attribute
df.info() #panda embedded function, summanry of the columns (name,#of entries, data type)
df.describe() #statistics on each column (count, mean, std, min, 25%...)
df = df.fillna(df.mean()) #makes all entries the mean of column (can be replaced with any var value)
df.dropna(inplace=True) #drop/delete rows/columns with null/NA values (many more use cases, https://www.geeksforgeeks.org/python-pandas-dataframe-dropna/)
df.drop('estimated_value', axis=1) #drop entire column (axis=1 means column)
df[['estimated_value', 'yearBuilt']] #takes a splice (two columns) of the df
df[(df.estimated_value<=800000) & (df.yearBuilt>2013)] #filter data with condition. condition returns boolean inside [], then takes estimated value of less than 800000
df.estimated_value.hist() #plot histogram of column
df.estimated_value.value_counts() #counts number in each unique value (like histogram)
df.zipcode.unique() #gets unique list of zipcode
df['priorSaleDate'] = pd.to_datetime(df.priorSaleDate) #converts var from object to date format
df.lastSaleDate.dt.month #displays all the month values (because it's date variable)
df['new_col']=1 # makes new col with name "new_col" and assigns all values as 1
df['diff']=df.lastSaleDate.dt.year-df.priorSaleDate.dt.year #computation b/w two columns & create new column
df.corr() #correlation b/w two columns
df.corr().loc['estimated_value',:].sort_values(ascending=False) # corrletion b/w estimated value and everything else (loc = location), and then sorts it from high to low
df2 = df.groupby(['zipcode','yearBuilt']).estimated_value.mean().reset_index() #assigns mean estimated value of all unique zipcodes and year built to df2. reset index converts type from series to dataframe
df3 = pd.merge(df,df2, on='zipcode', how='left') #joins df1 and df2 based on commons between zipcode column. how=left means using only keys/columns from the left frame. can also be left right outer innner
df.boxplot(column='estimated_value', by = 'zipcode') #multiple boxplots based on different zipcodes
 
#seaborn
sns.boxplot(df.estimated_value) #plots a column in df with boxplot format
sns.pairplot(df[['lastSaleAmount', 'estimated_value','zipcode']], hue = 'zipcode') #pairplot is for visualize relationship between two variables while multiple variables are present. creates figure with m*m subplots, histogram at diagonal. hue is legend
sns.stripplot(x=df.zipcode, y=df.estimated_value)  #like scatter plot but with discrete values on one axis
sns.violinplot(x=df.zipcode, y=df.estimated_value) #similar to stripplot but with thickness variance

#get datasets/example code from kaggle.com, sklearn, reddit..