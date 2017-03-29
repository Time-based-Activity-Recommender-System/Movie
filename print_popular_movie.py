import numpy as np
import pandas as pd

pop_cols = ['movie_id', 'movie_title']
pop_movies = pd.read_csv('ml-100k/popular.data', sep='\t', names=pop_cols, encoding='latin-1')
l = len(pop_movies) 
#print "Length = ", l
ind1 = np.random.randint(0, l)
ind2 = np.random.randint(0, l)
ind3 = np.random.randint(0, l)


while(ind1 == ind2):
	ind2 = np.random.randint(0, l)
while(ind1 == ind3 and ind2 == ind3):
	ind3 = np.random.randint(0, l)

print pop_movies['movie_title'][ind1]
print pop_movies['movie_title'][ind2]
print pop_movies['movie_title'][ind3]
