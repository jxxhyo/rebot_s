from django.urls import path
from . import views
from .views import csrf_token_view

urlpatterns = [
    path('', views.chatbot, name='chatbot'),
    path('login', views.login, name='login'),
    path('register', views.register, name='register'),
    path('logout', views.logout, name='logout'),
    path('csrf_token/', csrf_token_view, name='csrf_token'),
]