"""
URL configuration for Financial Sentiment Analysis
"""
from django.urls import path
from . import views

urlpatterns = [
    # Main dashboard
    path('', views.dashboard, name='dashboard'),
    path('dashboard/', views.dashboard, name='dashboard'),
    
    # API endpoints
    path('api/predict/', views.predict_sentiment, name='predict_sentiment'),
    path('api/batch/', views.batch_predict, name='batch_predict'),
    path('api/model-info/', views.model_info_api, name='model_info_api'),
    
    # Legacy URLs for compatibility
    path('home/', views.home, name='home'),
    path('ml-dashboard/', views.ml_dashboard, name='ml_dashboard'),
    path('api/predict_batch/', views.predict_batch_api, name='predict_batch_api'),
    path('model_info/', views.model_info, name='model_info'),
]
