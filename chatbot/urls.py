from django.urls import path
from . import views
from .views import set_csrf_cookie, RegisterUser


urlpatterns = [
    path('register/', RegisterUser.as_view(), name='register'), 
    path('chatbot/', views.chatbot, name='chatbot'), 
    path('csrf/', set_csrf_cookie, name='csrf')  # CSRF 쿠키를 설정하는 엔드포인트
]
