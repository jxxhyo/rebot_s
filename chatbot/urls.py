from django.urls import path
from . import views
from .views import set_csrf_cookie


urlpatterns = [
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('chatbot/', views.chatbot, name='chatbot'),
    path('set-csrf-cookie/', set_csrf_cookie, name='set_csrf_cookie'),
]
