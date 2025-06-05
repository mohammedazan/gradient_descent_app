from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_data, name='upload'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('preprocessing/', views.preprocessing, name='preprocessing'),
    path('algorithm/', views.algorithm, name='algorithm'),
    path('results/', views.results, name='results'),
]