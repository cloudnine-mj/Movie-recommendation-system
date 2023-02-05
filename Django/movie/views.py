from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings

import pandas as pd
import numpy as np
import urllib.request

from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

import warnings; warnings.simplefilter('ignore')

from django.contrib.auth.models import User
from django.contrib import auth


# Create your views here.

def index(request):
    return render(request, 'movie/index.html', {})


def signup(request):
    if request.method == "POST":
        if request.POST["password1"]==request.POST["password2"]:
            user = User.objects.create_user(
                username=request.POST["username"], password=request.POST["password1"])
            auth.login(request, user)
            return redirect('home')
        return render(request, 'signup.html')
    return render(request, 'movie/signup.html', {})


def login(request):
    return render(request, 'movie/login.html', {})




def pred_movie(request):
    return render(request, 'movie/pred_movie.html', {})


def movie_like(request):

    movie_name_input = request.POST['movie_name']
    print('=========================================')
    print(movie_name_input)
    print('=========================================')

    movie_data = pd.read_csv('media/crawling.csv')
    rating_mid = pd.read_csv('media/rating_mid.csv')
    cosine_sim_key = pd.read_csv('media/cosine_sim_key.csv')
    cosine_sim_act = pd.read_csv('media/cosine_sim_act.csv')
    cosine_sim_dir = pd.read_csv('media/cosine_sim_dir.csv')
    # cosine_sim_gen = pd.read_csv('media/cosine_sim_gen.csv')

    result_key = cosine_sim_key[movie_name_input].sort_values(ascending=False)[1:11].index
    result_act = cosine_sim_act[movie_name_input].sort_values(ascending=False)[1:11].index
    result_dir = cosine_sim_dir[movie_name_input].sort_values(ascending=False)[1:11].index
    # result_gen = cosine_sim_gen[movie_name_input].sort_values(ascending=False)[1:11].index

    key_recc = pd.Series()
    for item in result_key:
        key_recc = pd.concat([key_recc, movie_data[movie_data['title'] == item]])
    key_recommend = key_recc[['title', 'poster', 'link']]

    act_recc = pd.Series()
    for item in result_act:
        act_recc = pd.concat([act_recc, movie_data[movie_data['title'] == item]])
    act_recommend = act_recc[['title', 'poster', 'link']]

    dir_recc = pd.Series()
    for item in result_dir:
        dir_recc = pd.concat([dir_recc, movie_data[movie_data['title'] == item]])
    dir_recommend = dir_recc[['title', 'poster', 'link']]

    # gen_recc = pd.Series()
    # for item in result_gen:
    #     gen_recc = pd.concat([gen_recc, movie_data[movie_data['title'] == item]])
    # gen_recommend = gen_recc[['title', 'poster', 'link']]

    result_key_url = list(key_recommend['link'])
    result_key_image = list(key_recommend['poster'])
    result_key_poster = list(zip(result_key_url, result_key_image))
    result_key_names = list(key_recommend['title'])

    result_act_url = list(act_recc['link'])
    result_act_image = list(act_recc['poster'])
    result_act_poster = list(zip(result_act_url , result_act_image))
    result_act_names = list(act_recc['title'])

    result_dir_url = list(dir_recc['link'])
    result_dir_image = list(dir_recc['poster'])
    result_dir_poster = list(zip(result_dir_url, result_dir_image))
    result_dir_names = list(dir_recc['title'])

    # result_gen_url = list(gen_recc['link'])
    # result_gen_image = list(gen_recc['poster'])
    # result_gen_poster = list(zip(result_gen_url, result_gen_image))
    # result_gen_names = list(gen_recc['title'])


    movie_data.rename(columns = {'id': 'movieId'}, inplace = True)
    user_movie_mid = pd.merge(movie_data, rating_mid, on = 'movieId')

    movie_user_rating = user_movie_mid.pivot_table('rating', index = 'title', columns='userId').fillna(0)
    item_based_collabor = cosine_similarity(movie_user_rating)
    item_based_collabor = pd.DataFrame(data = item_based_collabor, index = movie_user_rating.index, columns = movie_user_rating.index)

    item_list = item_based_collabor[movie_name_input].sort_values(ascending=False)[1:11].index
    item_recc = pd.Series()
    for item in item_list:
        item_recc = pd.concat([item_recc, movie_data[movie_data['title'] == item]])
    item_recommend = item_recc[['movieId', 'title', 'crawling_title', 'poster', 'link']]

    result_movie_url = list(item_recommend['link'])
    result_movie_image = list(item_recommend['poster'])
    result_movie_poster = list(zip(result_movie_url, result_movie_image))

    result_movie_names = list(item_recommend['title'])


    context = {'result_key_poster':result_key_poster,
               'result_key_names':result_key_names,
               'result_act_poster':result_act_poster,
               'result_act_names':result_act_names,
               'result_dir_poster':result_dir_poster,
               'result_dir_names':result_dir_names,
               # 'result_gen_poster':result_gen_poster,
               # 'result_gen_names':result_gen_names,
               'result_movie_poster':result_movie_poster,
               'result_movie_names':result_movie_names}

    return render(request, 'movie/pred_movie_result.html', context)


def rating_movie(request):

    movie_data = pd.read_csv('media/crawling.csv')
    rating_data = pd.read_csv('media/ratings_small.csv').drop('timestamp', axis = 1)
    movie_data = movie_data.rename(columns = {'id': 'movieId'})

    top_movie_view = pd.DataFrame(columns=['movieId', 'title', 'popularity', 'poster', 'link'])

    for i in range(5):
        top_movie = movie_data[['movieId', 'title', 'popularity', 'poster', 'link']].sort_values('popularity', ascending=False).iloc[i]
        top_movie_view = top_movie_view.append(pd.DataFrame(top_movie).T, ignore_index=True)
    top_movie_view

    movie_title = list(top_movie_view['title'])
    movie_id = list(top_movie_view['movieId'])
    movie_poster = list(top_movie_view['poster'])

    title_id_image = list(zip(movie_title, movie_id, movie_poster))

    context = {'title_id_image':title_id_image}

    return render(request, 'movie/rating.html', context)


def rating(request):

    input_movie_ids = []
    input_user_ratings = []

    for input_name in request.POST.keys():
        if 'movie_id' in input_name:
            input_movie_ids.append(request.POST[input_name])
        elif 'user_rating' in input_name:
            input_user_ratings.append(request.POST[input_name])


    movie_data = pd.read_csv('media/crawling.csv')
    rating_data = pd.read_csv('media/ratings_small.csv').drop('timestamp', axis = 1)
    movie_data.rename(columns = {'id': 'movieId'}, inplace = True)

    user_id = 611
    uid_list = [user_id, user_id, user_id, user_id, user_id]
    new_user_rating = list(zip(uid_list, input_movie_ids, input_user_ratings))

    for i in new_user_rating:
        rating_data.loc[len(rating_data)] = i

    rating_data['userId'] = rating_data['userId'].astype('int')
    rating_data['movieId'] = rating_data['movieId'].astype('int')
    rating_data['rating'] = rating_data['rating'].astype('float')

    ratings_movies = pd.merge(movie_data, rating_data, on='movieId')

    user_rating_pivot = ratings_movies.pivot_table('rating', index = 'userId', columns = 'movieId').fillna(0)
    user_based_collabor = cosine_similarity(user_rating_pivot)
    user_based_collabor = pd.DataFrame(data = user_based_collabor, index = user_rating_pivot.index, columns = user_rating_pivot.index)

    user_sim_index = user_based_collabor.loc[user_id].sort_values(ascending=False).index[1]
    sim_user_rec = user_rating_pivot.loc[user_sim_index].sort_values(ascending=False)[:12].index
    user_movie_index = user_rating_pivot.loc[user_id][user_rating_pivot.loc[user_id] != 0.0].index
    sim_user_rec = [i for i in list(sim_user_rec) if i not in list(user_movie_index)]

    user_rec = movie_data.loc[sim_user_rec][['movieId', 'title', 'poster', 'link']]

    user_rec_url = list(user_rec['link'])
    user_rec_image = list(user_rec['poster'])
    user_rec_poster = list(zip(user_rec_url, user_rec_image))

    user_rec_names = list(user_rec['title'])


    # svd
    matrix = user_rating_pivot.values
    user_rating_mean = np.mean(matrix, axis=1)
    matrix_user_mean = matrix - user_rating_mean.reshape(-1, 1)
    user_mean = pd.DataFrame(matrix_user_mean, columns = user_rating_pivot.columns)

    U, sigma, Vt = svds(user_mean, k=12)
    sigma = np.diag(sigma)
    svd_user_predition_ratings = np.dot(np.dot(U, sigma), Vt) + user_rating_mean.reshape(-1, 1)
    svd_predition = pd.DataFrame(svd_user_predition_ratings, columns = user_rating_pivot.columns)

    user_row_number = user_id - 1
    sorted_pred = svd_predition.iloc[user_row_number].sort_values(ascending=False)
    user_data = rating_data[rating_data.userId == user_id]
    already_rated = user_data.merge(movie_data, on = 'movieId').sort_values('rating', ascending=False)

    predictions = movie_data[~movie_data.isin(already_rated['movieId'])]
    predictions = predictions.merge(pd.DataFrame(sorted_pred).reset_index(), on = 'movieId')
    predictions = predictions.rename(columns = {user_row_number : 'Predictions'}).sort_values('Predictions', ascending=False)

    pred_svd = predictions[['movieId', 'title', 'Predictions', 'poster', 'link']].head(10)

    svd_url = list(pred_svd['link'])
    svd_image = list(pred_svd['poster'])
    svd_poster = list(zip(svd_url, svd_image))

    svd_names = list(pred_svd['title'])
    svd_rating = list(pred_svd['Predictions'])
    # svd_pred = list(zip(svd_names, svd_rating))



    context = {'user_rec_poster':user_rec_poster,
               'user_rec_names':user_rec_names,
               'svd_poster':svd_poster,
               'svd_names':svd_names,
               'svd_rating':svd_rating}

    return render(request, 'movie/rating_result.html', context)
