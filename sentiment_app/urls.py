"""
URL configuration for sentiment_app.
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('demo/', views.demo, name='demo'),
    path('api/predict/', views.predict_single, name='predict_single'),
    path('api/predict_batch/', views.predict_batch, name='predict_batch'),
    path('api/model_info/', views.model_info, name='model_info'),
]
