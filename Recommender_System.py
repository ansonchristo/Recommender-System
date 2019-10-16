#!/usr/bin/env python
# coding: utf-8

# # This project is to test my basic understanding of recommending systems. My goal is to tell you what movies are most similar to your movie choice. 

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('white')


# In[3]:


column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)


# In[4]:


df.head()


# extracting movie title

# In[5]:


movie_titles = pd.read_csv("Movie_Id_Titles")
movie_titles.head()


# In[6]:


df = pd.merge(df,movie_titles,on='item_id')
df.head()


# Group by with avg rating

# In[7]:


df.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# Group by number of ratings for each movie

# In[8]:


df.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[9]:


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()


# Now set the number of ratings column:

# In[10]:


ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()


# Some visualizations...

# In[11]:


plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)


# In[12]:


plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)


# In[13]:


sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)


# ## Recommending Similar Movies

# This is definitely not surprising since not many people has watched all the movies

# In[14]:


moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()


# Most rated movie:

# In[15]:


ratings.sort_values('num of ratings',ascending=False).head()


# I decided to pick two of my favourite movies, star wars and Liar Liar

# In[16]:


ratings.head()


# Now let's grab the user ratings for those two movies:

# In[17]:


starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
starwars_user_ratings.head()


# Used corrwith to achieve a correlation between starwars, liar liar and the rest of the ratings

# In[21]:


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)
similar_to_starwars


# Remove all NAN values 

# In[20]:


corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()


# I started to realize this doesn't make sense since there are a lot of movies only watched once by users who also 
# watched starwars. I am going to filter out movies that have less than 100 ratings based on histogram visualization. 

# In[155]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# In[165]:


corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()


# Makes more sense! Empire Strikes Back, Return of the Jedi are related to Starwar movies 

# In[157]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()


# In[158]:


corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()






    

