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
ratings_data = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
 encoding='latin-1')
print type(ratings_data)
#print ratings
print ratings_data.shape
print ratings_data.head()

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

newuser_ratings = np.zeros(num_movies,dtype=np.uint8)

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

newuser_id = 944

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
ratings_data = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')

print ratings_data.tail()
print "Ratings_data length: ",len(ratings_data)

ratings = np.zeros((num_movies, num_users+1), dtype = np.uint8)
for i in range(len(ratings_data)):
	col = ratings_data['user_id'][i]-1
	row = ratings_data['movie_id'][i]-1
	ratings[row][col]=ratings_data['rating'][i]

#print "YPYO", ratings[204][716]	
did_rate = (ratings != 0) * 1

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

#ratings, ratings_mean = normalize_ratings(ratings, did_rate)
# Update some key variables now

num_users = ratings.shape[1]
num_features = 3

# Initialize Parameters theta (user_prefs), X (movie_features)

movie_features = np.random.randn( num_movies, num_features )
user_prefs = np.random.randn( num_users, num_features )
initial_X_and_theta = np.r_[movie_features.T.flatten(), user_prefs.T.flatten()]

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
	return np.r_[X_grad.T.flatten(), theta_grad.T.flatten()]


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

minimized_cost_and_optimal_params = optimize.fmin_cg(calculate_cost, fprime=calculate_gradient, x0=initial_X_and_theta, 								args=(ratings, did_rate, num_users, num_movies, num_features, reg_param), 								maxiter=100, disp=True, full_output=True ) 


# In[ ]:

cost, optimal_movie_features_and_user_prefs = minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]


# In[ ]:

# unroll once again

movie_features, user_prefs = unroll_params(optimal_movie_features_and_user_prefs, num_users, num_movies, num_features)
print user_prefs
# Make some predictions (movie recommendations). Dot product
all_predictions = movie_features.dot( user_prefs.T )
# add back the ratings_mean column vector to my (our) predictions
predictions_for_nikhil = all_predictions[:, 0:1] + ratings_mean
print predictions_for_nikhil
print newuser_ratings

