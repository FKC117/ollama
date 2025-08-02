from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload-success/', views.upload_success, name='upload_success'),
    path('process-dataset/', views.process_dataset, name='process_dataset'),
    path('results/', views.results, name='results'),
]