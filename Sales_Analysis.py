#!/usr/bin/env python
# coding: utf-8

# ## Sales Analysis

# #### Import necessary libraries

# In[1]:


import pandas as pd
import os


# Let's read a single csv file of the month of April and view the contents. 

# In[2]:


df = pd.read_csv('all_data.csv')
df.head(100)


# #### Task 1: Merge 12 months of sales data into a single CSV file
# 

# The dataset contains 12 different CSV files for 12 months of 2019. Let's now read all the files from the '/dataset' folder.

# In[3]:


filenames = [file for file in os.listdir('./dataset')]
print(*filenames, sep='\n')


# Now that we have read the filenames from the '/dataset' folder, we can concatenate into a single CSV file.

# In[4]:


all_data = pd.DataFrame()
for file in filenames:
    df = pd.read_csv('./dataset/'+file)
    all_data = pd.concat([all_data, df])

all_data.head(1000)


# In[5]:


# Saving all_data to a CSV file
all_data.to_csv('all_data.csv', index=False)


# We can read the merged and updated data directly from 'all_data.csv' so that we don't need to run the code above every time. 

# In[6]:


all_data = pd.read_csv('all_data.csv')
all_data.head()


# In[7]:


all_data.shape


# In[8]:


all_data.tail()


# In[9]:


all_data.describe()


# Clean up the data
# 
# 

# In[10]:


# Find rows with any NaN
any_nan_df = all_data[all_data.isna().any(axis=1)]
any_nan_df.head()


# In[11]:


any_nan_df.shape


# In[12]:


print('Looks like there are '+str(any_nan_df.shape[0])+' rows with atleast one NaN!')


# In[13]:


all_nan_df = all_data[all_data.isna().all(axis=1)]
all_nan_df.head()


# In[14]:


all_nan_df.shape


# In[15]:


print('And there are '+str(all_nan_df.shape[0])+' rows with all NaN\'s!')


# That means we can drop these rows as all of them have NaN's.

# #### Drop rows with NaN

# In[16]:


all_data = all_data.dropna(how='all')
all_data.head()


# In[17]:


all_data.shape


# #### Convert columns to correct type

# In[18]:


all_data.dtypes


# #### Data Cleanup Contd: Find 'Or' rows and delete them

# In[19]:


temp_df = all_data[all_data['Order Date'].str[0:2] == 'Or']
temp_df.head()


# In[20]:


temp_df.shape


# In[21]:


print('Looks like there are '+str(temp_df.shape[0])+' rows with the header row duplicated!')


# In[22]:


del temp_df


# Let's drop those rows now.

# In[23]:


all_data = all_data[all_data['Order Date'].str[0:2] != 'Or']
all_data.shape


# In[24]:


all_data['Quantity Ordered'] = pd.to_numeric(all_data['Quantity Ordered'])
all_data['Price Each'] = pd.to_numeric(all_data['Price Each'])
all_data.head()


# ### Augment data with additional columns

# #### Task 2: Create new 'Month' column from 'Order Date' column

# In[25]:


# Slicing first two characters from Order Date for month
all_data['Month'] = all_data['Order Date'].str[0:2]


# In[26]:


#Convert Month from str to int32
all_data['Month'] = all_data['Month'].astype('int32')
all_data.head()


# Okay, so we have successfully converted the 'Month' column from str to int.

# #### Task 3: Add a Sales column

# In[27]:


all_data['Sales'] = all_data['Quantity Ordered'] * all_data['Price Each']
all_data.head()


# #### Task 4: Add a City column

# In[28]:


# Let's use the .apply()
def get_city(address):
    return address.split(',')[1]

def get_state(address):
    return (address.split(',')[2]).split(' ')[1]

all_data['City'] = all_data['Purchase Address'].apply(lambda x: get_city(x)+' ('+get_state(x)+')')

all_data.head()


# ### Question 1: What was the best month for sales? How much was earned that month?

# In[29]:


results = all_data.groupby('Month').sum() 
results
# It only sums up the columns with numeric datatype!!


# Let's plot this!

# In[30]:


import  matplotlib.pyplot as plt


# In[31]:


months = range(1,13)
plt.figure(figsize=(9,4))
plt.bar(months,results['Sales'])
#plt.bar(results.index,results['Sales'])  <- Can use this too
plt.xticks(months)
plt.ylabel('Total Sales in a month in USD ($)')
plt.xlabel('Month Number')
plt.grid(axis='y')
plt.show()


# In[32]:


print(f"Thus, the best month in sales was Month {results.idxmax()['Sales']} with a sale of " + "${:,.2f}".format(results.max()['Sales']))


# So, December being the best month in sales followed by October, April and November can be explained by the festivities that are in these months. Generaly in December, shopping spends peak around Christmas and New Year's and is closely followed by the festivities in the other months respectively.

# ### Question 2: What city has the highest sales?

# In[33]:


results = all_data.groupby('City').sum() 
results


# In[34]:


cities = results.index
plt.figure(figsize=(9,4))
plt.bar(cities,results['Sales'])
#plt.bar(results.index,results['Sales'])  <- Can use this too
plt.xticks(cities, rotation='vertical', size=8)
plt.ylabel('Total Sales in a city in USD ($)')
plt.xlabel('City Name')
plt.grid(axis='y')
plt.show()


# ### Question 3: What time should we display advertisements to maximize likehood of customers buying products?

# First converting Order Date column from str to datetime object.

# In[35]:


all_data['Order Date'] = pd.to_datetime(all_data['Order Date'])


# In[36]:


all_data.head()


# In[37]:


all_data['Hour'] = all_data['Order Date'].dt.hour
all_data['Minute'] = all_data['Order Date'].dt.minute
all_data.head()


# In[38]:


results = all_data.groupby(['Hour']).count()
results


# In[39]:


hours = results.index
plt.plot(hours, results['Order ID'])
plt.xticks(hours)
plt.xlabel('Hour of the Day')
plt.ylabel('Total Number of Sales in an Hour')
plt.grid()
plt.show()


# From the chart above, it is clearly evident that the peaks in shopping occur around 12pm (1200 hrs) and 7pm (1900 hrs) across the entire 10 US cities. 

# ### Question 4: What products are most often sold together?

# In[40]:


all_data.head(10)


# Here, by carefully observing the data, we can say that if the Order ID of two or more rows match, the corresponding Products were sold together.

# In[41]:


# Keeping only the ones which have duplicated Order ID
df = all_data[all_data['Order ID'].duplicated(keep=False)]

# Joining all the Products with same Order ID by ',' and storing in 'Grouped' column
df['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))

# Take only the Order ID and Grouped columns and drop duplicates
df = df[['Order ID', 'Grouped']].drop_duplicates()
df.head(100)


# Now we can count number of occurences of the combinations.

# In[42]:


from itertools import combinations
from collections import Counter

count = Counter()

for row in df['Grouped']:
    row_list = row.split(',')
    count.update(Counter(combinations(row_list, 2)))

for key, value in count.most_common(10):
    print(key, value)


#  iPhone and Lightning Charging Cable are sold together the most 1005 times

# ### For more than 2 items taken at a time

# In[43]:


all_data.groupby('Order ID').count().sort_values(['Product'], axis=0, ascending=False)


# From the above table we can see that a particular Order ID occurs at most 5 times.

# In[44]:


from itertools import combinations
from collections import Counter

for comb in range(2, 6):
    count = Counter()
    print(f"\nTaking {comb} items at a time:")
    for row in df['Grouped']:
        row_list = row.split(',')
        count.update(Counter(combinations(row_list, comb)))
    
    for key, value in count.most_common(10):
        print(key, value)


# ### Question 5: What product sold the most? Why do you think it sold the most?

# In[45]:


product_group = all_data.groupby('Product')
quantity_ordered = product_group.sum()['Quantity Ordered']
quantity_ordered


# In[46]:


products = quantity_ordered.index

plt.bar(products, quantity_ordered)
plt.xticks(products, rotation='vertical', size=8)
plt.ylabel('Number of Units Sold')
plt.xlabel('Products')
plt.show()


# We can say that AAA Batteries (4-pack) were the most. This may be because the per unit price of this item is lowest. Let's see if we are correct!

# In[47]:


prices = all_data.groupby('Product').mean()['Price Each']
prices


# In[48]:


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(products, quantity_ordered, color='y')
ax1.set_xticklabels(products, rotation='vertical', size=8)
ax1.set_ylabel('Total Quantity Ordered', color='y')
ax1.set_xlabel('Products')

ax2.plot(products, prices, color='g')
ax2.set_ylabel('Mean Price ($)', color='g')
plt.show()


# So, we see an inverse correlation in the Quantity Ordered and Mean Prices. There are some inconsistencies such as Macbook Pro Laptop as greater price than LG Dryer but still Quantity Ordered is more for Mackbook Pro Laptop than for LG Dryer. This may be because demand is more for Macbook Pro Laptop than for LG Dryer.

# In[ ]:





# In[ ]:





# In[ ]:




