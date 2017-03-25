
# coding: utf-8
# In[16]:

import numpy as np
import pandas as pd

num_movies = 1682
num_users = 943


# read_csv using pandas. 
# Column names available in the readme file

#Reading users file(943*5):
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')
#print users
print users.shape       #i.e. dimensions: 943*5
print users.head()

#Reading ratings file(100000,4):
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
 encoding='latin-1')
print type(ratings)
#print ratings
print ratings.shape
print ratings.head()

#Reading items file(1682*24):
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')
#print items
print items.shape
print items.head()
print "\n\n"
print items['movie title'][0]
print items['movie title'][49]
print items['movie title'][70]
print items['movie title'][63]
print items['movie title'][68]
print items['movie title'][71]
print items['movie title'][81]
print items['movie title'][87]
print items['movie title'][93]
print items['movie title'][97]

#TODO: PUT ABOVE 10 IN MOVIERATINGS_UI_RUN.PY!!!!!!!!!!!!

# In[21]:
# create a logical matrix (matrix that represents whether a rating was made, or not)
# != is the logical not operator

did_rate = (ratings != 0) * 1


# In[22]:
#print did_rate


# In[23]:
# Here's what happens if we don't multiply by 1
#print (ratings != 0)


# In[24]:
#print (ratings != 0) * 1


# In[25]:
# Get the dimensions of a matrix using the shape property
#print ratings.shape


# In[26]:
#print did_rate.shape


# In[27]:
# Let's make some ratings. A 10 X 1 column vector to store all the ratings I make
newuser_ratings = np.zeros(num_movies,dtype=np.uint8)
#print newuser_ratings


# In[28]:

# Python data structures are 0 based

#print newuser_ratings[9] 


# In[29]:

# I rate 3 movies

#TODO: EXTRACT FROM MOVIERATINGS_UI_RUN.PY SLIDERS!!!!!!!!!!!!!
newuser_ratings[0] = 8
newuser_ratings[49] = 7
newuser_ratings[70] = 3
newuser_ratings[63] = 6
newuser_ratings[68] = 9
newuser_ratings[71] = 3
newuser_ratings[81] = 5
newuser_ratings[87] = 7
newuser_ratings[93] = 8
newuser_ratings[97] = 4

print newuser_ratings

#append in ml-100k/u.data

newuser_id = 999

# rat_mat = ratings['rating'][0]
# for i in range(1,1683):
# 	rat_mat.append(ratings['rating'][i])
# print "this"
# print ratings.tail()
#d is dictionary
d = {'user_id': [newuser_id],  'movie_id': [0], 'rating': [newuser_ratings[0]] , 'unix_timestamp':[800000000]}
df = pd.DataFrame(d)
print "Dataframe"
print df
# ratings = pd.concat([ratings,df])
# ratings.append(df)
df.to_csv('ml-100k/u.data',mode='a' ,sep='\t',index=False, header=False) 


d = {'user_id': [newuser_id],  'movie_id': [49], 'rating': [newuser_ratings[49]] , 'unix_timestamp':[800000001]}
df = pd.DataFrame(d)
df.to_csv('ml-100k/u.data',mode='a' ,sep='\t',index=False, header=False) 

d = {'user_id': [newuser_id],  'movie_id': [70], 'rating': [newuser_ratings[70]] , 'unix_timestamp':[800000002]}
df = pd.DataFrame(d)
df.to_csv('ml-100k/u.data',mode='a' ,sep='\t',index=False, header=False) 

d = {'user_id': [newuser_id],  'movie_id': [63], 'rating': [newuser_ratings[63]] , 'unix_timestamp':[800000003]}
df = pd.DataFrame(d)
df.to_csv('ml-100k/u.data',mode='a' ,sep='\t',index=False, header=False) 

d = {'user_id': [newuser_id],  'movie_id': [68], 'rating': [newuser_ratings[68]] , 'unix_timestamp':[800000004]}
df = pd.DataFrame(d)
df.to_csv('ml-100k/u.data',mode='a' ,sep='\t',index=False, header=False) 

d = {'user_id': [newuser_id],  'movie_id': [71], 'rating': [newuser_ratings[71]] , 'unix_timestamp':[800000005]}
df = pd.DataFrame(d)
df.to_csv('ml-100k/u.data',mode='a' ,sep='\t',index=False, header=False) 

d = {'user_id': [newuser_id],  'movie_id': [81], 'rating': [newuser_ratings[81]] , 'unix_timestamp':[800000006]}
df = pd.DataFrame(d)
df.to_csv('ml-100k/u.data',mode='a' ,sep='\t',index=False, header=False) 

d = {'user_id': [newuser_id],  'movie_id': [87], 'rating': [newuser_ratings[87]] , 'unix_timestamp':[800000007]}
df = pd.DataFrame(d)
df.to_csv('ml-100k/u.data',mode='a' ,sep='\t',index=False, header=False) 

d = {'user_id': [newuser_id],  'movie_id': [93], 'rating': [newuser_ratings[93]] , 'unix_timestamp':[800000008]}
df = pd.DataFrame(d)
df.to_csv('ml-100k/u.data',mode='a' ,sep='\t',index=False, header=False) 

d = {'user_id': [newuser_id],  'movie_id': [97], 'rating': [newuser_ratings[97]] , 'unix_timestamp':[800000009]}
df = pd.DataFrame(d)
df.to_csv('ml-100k/u.data',mode='a' ,sep='\t',index=False, header=False) 

'''
df.to_csv('ml-100k/u.data', sep = '\t')

l = [newuser_id, 0, newuser_ratings[0], 8000000000] #list
fd = open('ml-100k/u.data','a')
print l
s = ""
for i in l:
	s = s + str(i) + " "
s = s + '\n'
fd.write(s)
fd.close()'''

ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')
print ratings.tail()
#Now ratings are the new ratings

'''
# In[30]:
# Update ratings and did_rate

#rat_mat = append(newuser_ratings, rat_mat, axis = 1)
#print 'SONALI'
#did_rate = append(((newuser_ratings != 0) * 1), did_rate, axis = 1)

'''
'''
# In[31]:

print ratings


# In[32]:

ratings.shape


# In[33]:

did_rate


# In[34]:

print did_rate


# In[35]:

did_rate.shape


# In[36]:

# Simple explanation of what it means to normalize a dataset

a = [10, 20, 30]
aSum = sum(a)


# In[37]:

print aSum


# In[38]:

aMean = aSum / 3


# In[39]:

print aMean


# In[40]:

aMean = mean(a)
print aMean


# In[41]:

a = [10 - aMean, 20 - aMean, 30 - aMean]
print a


# In[42]:

print ratings
'''

# In[43]:

# a function that normalizes a dataset

def normalize_ratings(ratings, did_rate):
    num_movies = ratings.shape[0]
    
    ratings_mean = np.zeros(shape = (num_movies, 1))
    ratings_norm = np.zeros(shape = ratings.shape)
    
    for i in range(num_movies): 
        # Get all the indexes where there is a 1
        idx = np.where(did_rate[i] == 1)[0]
        #  Calculate mean rating of ith movie only from user's that gave a rating
        ratings_mean[i] = mean(ratings[i, idx])
        ratings_norm[i, idx] = ratings[i, idx] - ratings_mean[i]
    
    return ratings_norm, ratings_mean

        
    


# In[44]:

# Normalize ratings

ratings, ratings_mean = normalize_ratings(ratings, did_rate)

print ratings.tail()

# In[45]:

# Update some key variables now
'''
num_users = ratings.shape[1]
num_features = 3


# In[46]:

# Simple explanation of what it means to 'vectorize' a linear regression
'''
'''
X = array([[1, 2], [1, 5], [1, 9]])
Theta = array([[0.23], [0.34]])


# In[47]:

print X


# In[48]:

print Theta


# In[49]:

Y = X.dot(Theta)
print Y

'''

# In[50]:

# Initialize Parameters theta (user_prefs), X (movie_features)
'''
movie_features = random.randn( num_movies, num_features )
user_prefs = random.randn( num_users, num_features )
initial_X_and_theta = r_[movie_features.T.flatten(), user_prefs.T.flatten()]


# In[51]:

print movie_features


# In[52]:

print user_prefs


# In[53]:

print initial_X_and_theta


# In[54]:

initial_X_and_theta.shape


# In[55]:

movie_features.T.flatten().shape


# In[56]:

user_prefs.T.flatten().shape


# In[57]:

initial_X_and_theta


# In[58]:

def unroll_params(X_and_theta, num_users, num_movies, num_features):
	# Retrieve the X and theta matrixes from X_and_theta, based on their dimensions (num_features, num_movies, num_movies)
	# --------------------------------------------------------------------------------------------------------------
	# Get the first 30 (10 * 3) rows in the 48 X 1 column vector
	first_30 = X_and_theta[:num_movies * num_features]
	# Reshape this column vector into a 10 X 3 matrix
	X = first_30.reshape((num_features, num_movies)).transpose()
	# Get the rest of the 18 the numbers, after the first 30
	last_18 = X_and_theta[num_movies * num_features:]
	# Reshape this column vector into a 6 X 3 matrix
	theta = last_18.reshape(num_features, num_users ).transpose()
	return X, theta


# In[59]:

def calculate_gradient(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)
	
	# we multiply by did_rate because we only want to consider observations for which a rating was given
	difference = X.dot( theta.T ) * did_rate - ratings
	X_grad = difference.dot( theta ) + reg_param * X
	theta_grad = difference.T.dot( X ) + reg_param * theta
	
	# wrap the gradients back into a column vector 
	return r_[X_grad.T.flatten(), theta_grad.T.flatten()]


# In[60]:

def calculate_cost(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)
	
	# we multiply (element-wise) by did_rate because we only want to consider observations for which a rating was given
	cost = sum( (X.dot( theta.T ) * did_rate - ratings) ** 2 ) / 2
	# '**' means an element-wise power
	regularization = (reg_param / 2) * (sum( theta**2 ) + sum(X**2))
	return cost + regularization


# In[64]:

# import these for advanced optimizations (like gradient descent)

from scipy import optimize


# In[65]:

# regularization paramater

reg_param = 30


# In[67]:

# perform gradient descent, find the minimum cost (sum of squared errors) and optimal values of X (movie_features) and Theta (user_prefs)

minimized_cost_and_optimal_params = optimize.fmin_cg(calculate_cost, fprime=calculate_gradient, x0=initial_X_and_theta,args=(ratings, did_rate, num_users, num_movies, num_features, reg_param), 								maxiter=100, disp=True, full_output=True ) 


# In[ ]:

cost, optimal_movie_features_and_user_prefs = minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]


# In[ ]:

# unroll once again

movie_features, user_prefs = unroll_params(optimal_movie_features_and_user_prefs, num_users, num_movies, num_features)


# In[ ]:

print movie_features


# In[ ]:

print user_prefs


# In[ ]:

# Make some predictions (movie recommendations). Dot product

all_predictions = movie_features.dot( user_prefs.T )


# In[ ]:

print all_predictions


# In[ ]:

# add back the ratings_mean column vector to my (our) predictions

predictions_for_nikhil = all_predictions[:, 0:1] + ratings_mean


# In[ ]:

print predictions_for_nikhil


# In[ ]:

print newuser_ratings



'''