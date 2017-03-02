import graphlab

# Install graphlab and open jupyter notebook  
# graphlab.get_dependencies()  # dependencies will get downloaded and installed in the installation director  
# Restart graphlab  
# import graphlab  

train_data = graphlab.SFrame(ratings_base)
test_data = graphlab.SFrame(ratings_test)
popularity_model = graphlab.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')
