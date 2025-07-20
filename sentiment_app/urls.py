"""
Enhanced URL configuration for sentiment_app.
"""
from django.urls import path
from . import views

urlpatterns = [
    # Main enhanced dashboard - ML focused
    path('', views.ml_dashboard, name='ml_dashboard'),
    
    # Legacy home page
    path('home/', views.home, name='home'),
    
    # API endpoints
    path('api/predict/', views.predict_api, name='predict_api'),
    path('api/predict_batch/', views.predict_batch_api, name='predict_batch_api'),
    path('model_info/', views.model_info, name='model_info'),
    path('model_history/', views.model_history, name='model_history'),
    path('retrain/', views.retrain_model, name='retrain_model'),
    path('load_historical_model/', views.load_historical_model, name='load_historical_model'),
]
