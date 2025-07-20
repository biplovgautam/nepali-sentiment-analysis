"""
URL configuration for sentiment_project project.
"""
from django.urls import path, include

urlpatterns = [
    path('', include('sentiment_app.urls')),
]
