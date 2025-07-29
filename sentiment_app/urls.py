"""
URL configuration for Financial Sentiment Analysis - Clean and Simple
"""
from django.urls import path
from . import views

urlpatterns = [
    # Main dashboard
    path('', views.dashboard, name='dashboard'),
    
    # API endpoints
    path('api/predict/', views.predict_sentiment, name='predict_sentiment'),
    path('api/batch/', views.batch_predict, name='batch_predict'),
    path('api/model-info/', views.model_info_api, name='model_info_api'),
]
