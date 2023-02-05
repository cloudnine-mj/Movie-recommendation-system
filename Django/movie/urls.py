from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views

app_name = 'movie'

urlpatterns = [
    path('', views.index, name='index'),

    path('pred_movie/', views.pred_movie, name='pred_movie'),
    path('pred_movie/predict/', views.movie_like, name='movie_like'),
    path('rating/', views.rating_movie, name='rating_movie'),
    path('rating/predict', views.rating, name='rating'),
    path('signup/', views.signup, name = 'signup'),
    path('login/', views.login, name = 'login'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
