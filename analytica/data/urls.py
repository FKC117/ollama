from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('process-dataset/', views.process_dataset, name='process_dataset'),
    path('ai-chat/', views.ai_chat, name='ai_chat'),
]