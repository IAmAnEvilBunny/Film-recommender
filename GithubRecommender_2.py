# -*- coding: utf-8 -*-
"""
Recommender System

Created on Thu Apr 16 22:25:47 2020

@author: Lawrence Arscott

for use with dataset compiled by GroupLens Research
https://grouplens.org/datasets/movielens/

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
    History and Context.
    ACM Transactions on Interactive Intelligent Systems (TiiS)
    5, 4: 19:1–19:19.
    https://doi.org/10.1145/2827872

"""

# In[1]:

import numpy as np

import random

import scipy
from scipy import optimize

from tqdm import tqdm
# In[2]:


def specify_my_ratings(n_films, Films):
    # My ratings
    my_ratings = np.zeros(n_films)
    my_ratings[1] = 4
    my_ratings[7] = 3
    my_ratings[12] = 5
    my_ratings[54] = 4
    my_ratings[64] = 5
    my_ratings[66] = 3
    my_ratings[69] = 5
    my_ratings[98] = 2
    my_ratings[183] = 4
    my_ratings[226] = 5
    my_ratings[355] = 5

    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            print('Rated {} for {}\n'.format(my_ratings[i], Films[i]))

    return my_ratings
# In[3]:


def main(movie_file, ratings_file, n_features, reg_lambda=10):
    """Recommends films based on your ratings and those of other users

    Optional extended description

    Parameters
    ----------
    movie_file : str
        Name of file containing film names and ids
    ratings_file : str
        Name of file containing ratings
    n_features : int
        Number of features associated with each film to be optimised
    reg_lambda : int
        Regularisation parameter. Prevents overfitting.

    Returns
    -------
    Films : numpy.ndarray of strings
        Array containing film names
    ids : numpy.ndarray of int
        ids[i] is the film id corresponding to the film with title Films[i]
    predictions : numpy.ndarray of floats
        Entry [i, j] is the predicted score for user j for the film with
        title Films[i]

    """
    # Format data, see format_data for details
    Films, ids, ratings = read_files(movie_file, ratings_file)

    # Useful values
    n_films = len(Films)  # Number of different films
    users = int(ratings[-1, 0]) + 1  # Number of users
    print('Initial data contains ratings for {} films across {} users'.format(
            n_films, users))

    # Reindex films
    ratings[:, 1] = reindex(n_films, ratings[:, 1], ids)

    # Divide data into training set, cross validation set and test set
    ratings, cv_ratings, test_ratings = divide_data(ratings, 20, 10)

    # Add in my ratings, I will be user 0, this also means we can start
    # indexing users from 0
    my_ratings = specify_my_ratings(n_films, Films)
    i_rated = np.where(my_ratings != 0)[0]  # Indices of films I rated

    # Add a column of zeros to specify I am user 0, and add film indices
    my_ratings = np.c_[np.zeros(len(i_rated)), i_rated, my_ratings[i_rated]]

    # Add my ratings to the ratings matrix
    ratings = np.r_[my_ratings, ratings]

    # Obtain R_mat and rating_mat (see function for details)...
    # ...for training set...
    print('Creating training set rating matrices...')
    R_mat, rating_mat = R_and_rating_mat(ratings, n_films, users)
    del ratings  # Clear up memory space

    # ...and for cv set
    print('Creating training set rating matrices...')
    cv_R_mat, cv_rating_mat = R_and_rating_mat(cv_ratings, n_films, users)
    del cv_ratings  # Clear up memory space

    # Compress data (remove data corresponding to films with very few ratings)
    Films, ids, R_mat, rating_mat, cv_R_mat, cv_rating_mat, n_films =\
        compress(R_mat, rating_mat, cv_R_mat, cv_rating_mat, Films, ids)

    # Normalise the ratings matrix and obtain the mean score for each film,
    # as well as the total number of people who rated each film
    rating_mat, means_mat, rated_sums = normalise_ratings(R_mat, rating_mat)

    # Initialise parameters (feature matrix and theta)
    parameters0 = param_init(n_films, n_features, users)

    # Optimise parameters
    Optimised_params = \
        scipy.optimize.minimize(cost,
                                parameters0,
                                args=(rating_mat,
                                      R_mat, users, n_films,
                                      n_features, reg_lambda),
                                method='TNC',
                                jac=vec_gradients,
                                tol=2)

    # Extract X and theta matrices
    X_opt, theta_opt = unpack(Optimised_params.x, users, n_films, n_features)

    # Add mean scores back to the previously normalised rating_mat
    rating_mat += means_mat * R_mat

    # Obtain our predictions
    predictions = X_opt @ theta_opt.T + means_mat

    # Assess how well predictions work on the training set
    print('Mse of {} on the training set'.format(
            mean_squared_error(predictions, R_mat, rating_mat, True)))
    print('Compared to {} when simpling using mean scores'.format(
            mean_squared_error(means_mat, R_mat, rating_mat, True)))

    # Assess how well predictions work on the cross validation set
    print('Mse of {} on the cv set'.format(
            mean_squared_error(predictions, cv_R_mat, cv_rating_mat, True)))
    print('Compared to {} when simpling using mean scores'.format(
            mean_squared_error(means_mat, cv_R_mat, cv_rating_mat, True)))

    return X_opt, theta_opt, Films, means_mat, R_mat, rating_mat, \
        cv_rating_mat, cv_R_mat, predictions, ids
# In[5]:


def compress(R_mat, rating_mat, cv_R_mat, cv_rating_mat, Films, ids,
             min_ratings=4):
    # Compress data
    # Remonves entries in R_mat and rating_mat corresponding to films with
    # less then min_ratings total ratings
    print('compressing')

    # Select films to be removed
    n_ratings = np.sum(R_mat, 1)

    # Obtain indices of rejected films
    rejected = np.where(n_ratings < min_ratings)

    # Remove data correponding to rejected films
    R_mat = np.delete(R_mat, rejected, axis=0)
    rating_mat = np.delete(rating_mat, rejected, axis=0)
    cv_R_mat = np.delete(cv_R_mat, rejected, axis=0)
    cv_rating_mat = np.delete(cv_rating_mat, rejected, axis=0)
    Films = np.delete(Films, rejected, axis=0)
    ids = np.delete(ids, rejected, axis=0)

    print('{} films with less than {} ratings rejected'.format(
            len(rejected[0]), min_ratings))

    n_films = len(Films)  # Update n_films

    return Films, ids, R_mat, rating_mat, cv_R_mat, cv_rating_mat, n_films


def cost(Parameters, Y, R, users, films, features, reg_lambda):
    # Cost function : this is the function we will be minimising

    # Unpack parameters
    X, theta = unpack(Parameters, users, films, features)

    # Calculate errors squared term
    errors2 = (X @ theta.T * R - Y)**2

    # Calculate regularisation terms
    reg_theta = sum(sum(theta**2))
    reg_X = sum(sum(X**2))

    # Putting it all together
    J = 1/2 * sum(sum(errors2)) + reg_lambda/2 * (reg_theta + reg_X)

    # Track progress
    print(J)

    return J


def divide_data(ratings, cv_pc, test_pc):
    # Divide data into training set, cross validation set and test set
    # according to percentages given
    n_ratings = len(ratings)
    n_cv = (n_ratings*cv_pc) // 100
    n_test = (n_ratings*test_pc) // 100

    # Printing
    print('dividing data')

    # Create cross-validation set
    # Obtain a random set of indices
    cv_indices = random.sample(range(n_ratings), n_cv)

    # Take out rows in ratings correponding to the cv indices and assign
    # to cv_ratings
    cv_ratings = ratings[cv_indices, :]
    ratings = np.delete(ratings, cv_indices, axis=0)

    # Update n_ratings
    n_ratings = len(ratings)

    # Repeat process with test indices
    test_indices = random.sample(range(n_ratings), n_test)
    test_ratings = ratings[test_indices, :]
    ratings = np.delete(ratings, test_indices, axis=0)

    print('Created a training set with {} ratings and a cross-validation\
          set with {} ratings'.format(n_ratings, n_cv))

    return ratings, cv_ratings, test_ratings


def gradients(Parameters, Y, R, users, n_films, n_features, reg_lambda):
    # Calculate gradients
    # No longer used, there is a vectorised version

    # Unpack parameters
    X, theta = unpack(Parameters, users, n_films, n_features)

    # Calculate X gradients
    X_grad = np.zeros((n_films, n_features))

    for i in tqdm(range(n_films)):
        # Get indices of users which have seen film i
        user_idx = np.where(R[i, :] == 1)

        # Create temporary Theta and Y matrices containing only information
        # relevant to users who have seen movie i
        theta_user = np.zeros((len(user_idx[0]), n_features))
        theta_user[:, :] = theta[user_idx, :]
        Y_user = Y[i, user_idx]

        # Calculate row i of X gradients
        X_grad[i, :] = (np.dot(X[i, :], theta_user.T) - Y_user) @ theta_user

    #  Calculate theta gradients
    Theta_grad = np.zeros((users, n_features))

    for i in range(users):
        # Get indices of movies user i has seen
        movie_idx = np.where(R[:, i] == 1)

        # Create temporary X and Y matrices containing only information
        # relevant to movies seen by user i
        X_movies = np.zeros((len(movie_idx[0]), n_features))
        X_movies[:, :] = X[movie_idx, :]
        Y_movies = Y[movie_idx, i]

        # Calculate row i of theta gradients
        Theta_grad[i, :] = (np.dot(theta[i, :], X_movies.T)
                            - Y_movies) @ X_movies

        X_grad = X_grad + reg_lambda*X
        Theta_grad = Theta_grad + reg_lambda*theta

    gradients = pack(X_grad, Theta_grad)

    return gradients


def mean_squared_error(prediction_mat, R_mat, rating_mat, bounds=False):
    # Calculates the mean squared error of the predictions we obtain
    n_ratings = np.sum(R_mat)

    if bounds:
        # Max rating is 5 stars, min is half a star
        prediction_mat[prediction_mat > 5] = 5
        prediction_mat[prediction_mat < 0.5] = 0.5

    errors2 = ((prediction_mat - rating_mat)*R_mat)**2
    mse = np.sum(errors2) / n_ratings

    return mse


def normalise_ratings(R_mat, rating_mat):
    # Normalise ratings so that the mean score for each film is 0
    # Also returns the mean score for each film, as well as the total number
    # of people who saw each film

    score_sums = np.sum(rating_mat, 1)  # Total score for each film
    rated_sums = np.sum(R_mat, 1)  # Total number of people who rated each film
    means = score_sums / rated_sums  # Mean score for each film
    means[np.isnan(means)] = 0  # In case any films had 0 ratings

    # Subtract mean score from any film that was rated
    rating_mat[R_mat == 0] = np.nan
    means_mat = (np.tile(means, (len(R_mat[0]), 1))).T
    rating_mat -= means_mat
    rating_mat[np.isnan(rating_mat)] = 0

    return rating_mat, means_mat, rated_sums


def pack(X, theta):
    # Packs all parameters into a single vector to pass on to the minimisation
    # function
    n_X = X.size
    n_theta = theta.size

    Parameters = np.zeros((n_X + n_theta, 1))
    Parameters[0:n_X] = X.reshape(n_X, 1)
    Parameters[n_X:] = theta.reshape(n_theta, 1)

    return Parameters


def param_init(n_films, n_features, users):
    # Initialise feature matrix and theta, return as a single vector
    film_features0 = np.random.random((n_films, n_features))
    theta0 = np.random.random((users, n_features))

    return pack(film_features0, theta0)


def R_and_rating_mat(ratings, n_films, users):
    # Obtain Rated from ratings. This is a matrix of integers where value
    # [i, j] corresponds to user j and film i

    # Start user indexing from 0, and set values to be integers
    Rated = ratings[:, 0:2].astype(int)

    # Create indicator matrix R_mat
    # R_mat[i, j] = 1 if user j has seen film i
    R_mat = np.zeros((n_films, users))
    columns = Rated[:, 0]
    rows = Rated[:, 1]
    R_mat[rows, columns] = 1

    # Create rating matrix rating_mat
    # rating_mat[i, j] is the rating user j gave to film i
    rating_mat = np.zeros((n_films, users))
    for i in tqdm(range(len(Rated))):
        rating_mat[Rated[i, 1], Rated[i, 0]] = ratings[i, 2]

    return R_mat, rating_mat


def read_files(movie_file, ratings_file):
    """Converts csv files into matrices we can use for optimisation

    This function has been tailored to read in data files made available by
    GroupLens research on their website
    https://grouplens.org/datasets/movielens/
    The function reads in the 'movies.csv' and the 'ratings.csv' files

    Parameters
    ----------
    movie_file : str
        Name of spreadsheet containing film names.
        Column 1 contains the film's id
        Column 2 contaiins the film's title
        [Unused] Column 3 contains the film's genres

    ratings_file : str
        Name of spreadsheet containing ratings.
        Column 1 contains the user id
        Column 2 contains the film's id
        Column 3 contains the film's rating
        [Unused] Column 4 contains the timestamp

    Returns
    -------
    Films : numpy.ndarray of strings
        Contains the films' titles

    ids : numpy array of ints
        ids[i] is the id corresponding to the film with title Films[i]

    ratings : numpy array of floats
        Rating information corresponding to columns 1-3 of rating_file

    """
    # Read in film names and ids
    file = open(movie_file, encoding="utf8")
    Films = np.loadtxt(file, delimiter=',', dtype='str', skiprows=1,
                       usecols=(1))

    file = open(movie_file, encoding="utf8")
    ids = np.loadtxt(file, delimiter=',', dtype='int', skiprows=1, usecols=(0))

    # Read in ratings
    # Column 0 indicates user index, column 1 the film index and column 2
    # indicates rating, on a scale from 0 to 5.
    file = open(ratings_file, encoding="utf8")
    print('Reading in ratings...')
    ratings = np.loadtxt(file, delimiter=',', dtype='float', usecols=(0, 1, 2),
                         skiprows=1, max_rows=3000000)
    print('Done !')

    return Films, ids, ratings


def reindex(n_films, old_indices, ids):
    # Reindexes films from their film-id to the index of the array ids
    # corresponding to their id, so that every film has an id between 0 and
    # n_films-1
    # Faster version I stole from stack exchange

    values = np.arange(n_films)  # Matching array of new indices for ids
    mapping_ar = np.zeros(ids.max()+1, dtype=values.dtype)
    mapping_ar[ids] = values

    # Entry i in mapping_ar correponds to the value we'd like to replace
    # entries in old_indices equal to i with

    new_indices = mapping_ar[old_indices]

    return new_indices


def old_reindex(n_films, old_indices, ids):
    # Reindexes films from their id to a number between 0 and n_films-1
    # There is probably a much better way to do this
    print('reindexing ' + str(n_films) + ' films')

    for index in tqdm(range(n_films)):
        old_indices[old_indices == ids[index]] = index

    return old_indices


def unpack(Parameters, users, n_films, n_features):
    # Unpacks parameter vector into X and theta matrices
    n_X = n_films * n_features
    X = Parameters[0:n_X].reshape(n_films, n_features)
    theta = Parameters[n_X:].reshape(users, n_features)

    return X, theta


def vec_gradients(Parameters, Y, R, users, n_films, n_features, reg_lambda):
    # Vectorised version of gradient calculation

    # Unpack parameters
    X, theta = unpack(Parameters, users, n_films, n_features)

    X_grad = np.zeros((n_films, n_features))
    Theta_grad = np.zeros((users, n_features))

    X_grad = (X @ theta.T - Y)*R @ theta
    Theta_grad = ((X @ theta.T - Y)*R).T @ X

    # Add regularisation term
    X_grad = X_grad + reg_lambda*X
    Theta_grad = Theta_grad + reg_lambda*theta

    # return as a single vector
    gradients = pack(X_grad, Theta_grad)

    return gradients


# In[6]:
X_opt, theta_opt, Films, means_mat, R_mat, rating_mat, cv_rating_mat, cv_R_mat, predictions, ids = \
        main('movies.csv', 'ratings.csv', 10, 10)
