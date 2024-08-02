from django.urls import path
from . import views

urlpatterns = [
    path('', views.find_similar_faces, name='name'),  # Main view for uploading and finding similar faces
]
