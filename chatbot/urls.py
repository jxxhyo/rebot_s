from django.urls import path
from . import views
from .views import set_csrf_cookie

urlpatterns = [
    path('', views.chatbot, name='chatbot'),
    path('login', views.login, name='login'),
    path('register', views.register, name='register'),
    path('logout', views.logout, name='logout'),
    path('csrf/', set_csrf_cookie, name='csrf')  # CSRF 쿠키를 설정하는 엔드포인트
]