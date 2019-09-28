import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]

##Step 1: Read CSV File
#df = dataframe
df = pd.read_csv("movie_dataset.csv")
#print(df.head())
print(df.columns)

##Step 2: Select Features
features = ['keywords','cast','genres','director']
#keywords has NAN convert it to empty string

for feature in features:
	df[feature] = df[feature].fillna('')
##Step 3: Create a column in DF which combines all selected features
def combine_features(row):
	return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']

df["combined_features"] = df.apply(combine_features,axis=1)
#print(df["combined_features"])

##Step 4: Create count matrix from this new combined column
cv = CountVectorizer()
#print(count_matrix)
count_matrix = cv.fit_transform(df["combined_features"])


##Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)
#print(cosine_sim)
#/movie_user_likes = "Batman"
a=1
while(a==1):
	movie_user_likes = input("Enter a movie which you like: ")
	movie_user_likes = str(movie_user_likes)
	#print("movie ",movie_user_likes)

	## Step 6: Get index of this movie from its title
	movie_index = get_index_from_title(movie_user_likes)
	similar_movies = list(enumerate(cosine_sim[movie_index]))

	## Step 7: Get a list of similar movies in descending order of similarity score
	#rev since most similar should come first and sort with second index hence [1]
	sorted_movie_list = sorted(similar_movies,key = lambda x:x[1],reverse=True)

	new=[]
	for i in range(len(sorted_movie_list)):
		#print(sorted_movie_list[i])
		#print(sorted_movie_list[i][1])
		if(sorted_movie_list[i][1]>0.3):
			new.append(sorted_movie_list[i])
	print("Modified List ",new)
	i = 0
	for movie in new:
		recommended_movie=get_title_from_index(movie[0])
		#print(movie,recommended_movie)
		if recommended_movie!=movie_user_likes:
			print(recommended_movie," cosine similarity: ",movie[1])

	a = int(input("Continue? (1/0) "))
