# -*- coding: utf-8 -*

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import os

# Check if the folder already exists so you don't redownload every run
if not os.path.exists("ml-100k"):
    # Download the dataset
    subprocess.run(["wget", "-q", "https://files.grouplens.org/datasets/movielens/ml-100k.zip"], check=True)
    
    # Extract the dataset
    subprocess.run(["unzip", "-q", "ml-100k.zip"], check=True)
    
    # Optional: Remove the zip file to save space
    subprocess.run(["rm", "ml-100k.zip"], check=True)
    
ratings=pd.read_csv('ml-100k/u.data',sep='\t',
names=['user_id','movie_id','rating','timestamp'])

movies=pd.read_csv('ml-100k/u.item',
sep='|',
encoding='latin-1',
header=None,
usecols=[0,1],
names=['movie_id','title'])

df=ratings.merge(movies,on='movie_id')
df.head(50)

print(df.shape , "\n")
print(df.isnull().sum(), "\n")
print(df['rating'].describe())

plt.figure(figsize=(6,4))
df['rating'].hist(bins=5)
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

top=df['title'].value_counts().head(10)
plt.figure(figsize=(10,5))
top.sort_values().plot(kind='barh')
plt.title('Top 10 Most Rated Movies')
plt.show()

movie_matrix=df.pivot_table(index='user_id',
columns='title',
values='rating')
movie_matrix.head()

movie_similarity=cosine_similarity(movie_matrix.fillna(0).T)
similarity_df=pd.DataFrame(movie_similarity,
index=movie_matrix.columns,
columns=movie_matrix.columns)
similarity_df.iloc[:5,:5]

pearson_correlation = movie_matrix.corr(method='pearson')
pearson_correlation.head()

def recommend_pearson(movie_name, n=10):
    scores = pearson_correlation[movie_name].sort_values(ascending=False)
    return scores.iloc[1:n+1]

movie_for_comparison = 'Star Wars (1977)'
rec_pearson = recommend_pearson(movie_for_comparison)

plt.figure(figsize=(8,5))
rec_pearson.sort_values().plot(kind='barh')
plt.title(f'Top Recommendations for {movie_for_comparison} (Pearson Correlation)')
plt.xlabel('Pearson Correlation')
plt.show()

plt.figure(figsize=(10,8))
subset=similarity_df.iloc[:20,:20]
plt.imshow(subset,cmap='viridis')
plt.title('Movie Similarity Heatmap (20 Movies)')
plt.colorbar()
plt.xticks(range(20),subset.columns,rotation=90,fontsize=6)
plt.yticks(range(20),subset.index,fontsize=6)
plt.show()

def recommend(movie_name,n=10):
    scores=similarity_df[movie_name].sort_values(ascending=False)
    return scores.iloc[1:n+1]

recommend('Legends of the Fall (1994)')

movie='Star Wars (1977)'
rec=recommend(movie)
plt.figure(figsize=(8,5))
rec.sort_values().plot(kind='barh')
plt.title(f'Top Recommendations for {movie}')
plt.xlabel('Cosine Similarity')
plt.show()

