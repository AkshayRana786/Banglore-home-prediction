#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)


# In[2]:


#Importing Dataset
df1 = pd.read_csv("House_Dataset.csv")
df1.head()


# In[3]:


df1.shape


# In[4]:


#Grouping and counting
df1.groupby('area_type')['area_type'].agg('count')


# In[5]:


# Drop columns
df2 = df1.drop(['area_type','society','balcony','availability'], axis = 'columns')
df2.head()


# In[6]:


# Search and sum of NaN values
df2.isnull().sum()


# In[7]:


# Drop NaN values
df3 = df2.dropna()
df3.isnull().sum()


# In[8]:


# Show Unique Values
df3['size'].unique()


# In[9]:


# Craete New column to store int number and splite the text from it.
df3['BHK'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))


# In[10]:


df3.head()


# In[11]:


# Show Unique Values
df3['BHK'].unique()


# In[12]:


# show BHK more then 20
df3[df3['BHK'] > 20]


# In[13]:


# Show Unique Values
df3['total_sqft'].unique()


# In[14]:


# Create function to convert float
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[15]:


# Check data is float or not
df3[~df3['total_sqft'].apply(is_float)].head(10)


# In[16]:


# take inpute the string and return avg int value.
def convert_sqft_to_num(x):
    token = x.split('-')
    if len(token) == 2:
        return (float(token[0]) + float(token[1])) / 2
    try:
        return float(x)
    except:
        return None


# In[17]:


convert_sqft_to_num('2100 - 2850')


# In[18]:


# create copy of dataset and replace it.
df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4.head()


# In[20]:


# Take location
df4.loc[410]


# In[28]:


# create copy of dataset.
df5 = df4.copy()
df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']
df5.head()


# In[29]:


# Find unique location
len(df5['location'].unique())


# In[32]:


#Find location as group by and sort it.
df5['location'] = df5['location'].apply(lambda x: x.strip())

location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending = False)
location_stats


# In[34]:


# Find location which is less then 10.
len(location_stats[location_stats <= 10])


# In[35]:


# Assign a variable
location_stats_less_then_10 = location_stats[location_stats <= 10]
location_stats_less_then_10


# In[36]:


# Conver location as Other location which location_stats less then 10.
df5['location'] = df5['location'].apply(lambda x: "other" if x in location_stats_less_then_10 else x)
len(df5['location'].unique())


# In[37]:


df5.head(10)


# ### 4th video Outlier Removal

# In[38]:


# Search data error like badroom is less then 300ft.
df5[df5['total_sqft'] / df5['BHK'] < 300]


# In[39]:


df5.shape


# In[42]:


# Copy dataset which has not less then 300ft per room.
df6 = df5[~(df5['total_sqft'] / df5['BHK'] < 300)]
df6.shape


# In[50]:


# Find price per sqft.
df6['price_per_sqft'].describe()


# In[51]:


# Remove data which price per sqft is lower and higher.
def remove_pps_outliners(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_out = pd.concat([df_out,reduced_df], ignore_index = True)
    return df_out

df7 = remove_pps_outliners(df6)
df7.shape


# In[58]:


# Create a visualizing chart for visualize data by location and dataset.
def plot_scatter_chart(df,location):
    bhk2 = df[(df['location'] == location) & (df['BHK'] == 2)]
    bhk3 = df[(df['location'] == location) & (df['BHK'] == 3)]
    matplotlib.rcParams["figure.figsize"] = (15,10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")


# In[61]:


# We cn remove those 2BHK apartment whose price_per_Sqft is less than mean price_per_sqft of 1BHK apartment

def remove_bhk_outliners(df):
    exclude_indices = np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk,bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean' : np.mean(bhk_df['price_per_sqft']),
                'std' : np.std(bhk_df['price_per_sqft']),
                'count' : bhk_df.shape[0]            
            }
        for bhk,bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices,bhk_df[bhk_df["price_per_sqft"] < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')

df8 = remove_bhk_outliners(df7)
df8.shape


# In[63]:


plot_scatter_chart(df8,'Hebbal')


# In[68]:


matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft, rwidth = 0.8)
plt.xlabel("Price Per Square Feet Area")
plt.ylabel("Count")


# In[69]:


df8['bath'].unique()


# In[71]:


df8[df8['bath'] > 10]


# In[73]:


plt.hist(df8['bath'], rwidth = 0.8)
plt.xlabel("Number of bathroom")
plt.ylabel("Count")


# In[75]:


# Outlier is badroom + 2 = bathroom
df8[df8['bath'] > df8['BHK'] + 2]


# In[76]:


# Remove the data
df9 = df8[df8['bath'] < df8['BHK'] + 2]
df9.shape


# In[78]:


# drop unnecessury field
df10 = df9.drop(["size","price_per_sqft"],axis = 1)
df10.head()


# In[80]:


# Save new dataset as csv file.
df10.to_csv("New_House_Dataset.csv",index=False)


# In[ ]:




