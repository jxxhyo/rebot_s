from django.urls import path
from .views import register, login_view, logout_view, chatbot, get_user_info, set_csrf_cookie

urlpatterns = [
    path('register/', register, name='register'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('chatbot/', chatbot, name='chatbot'),
    path('get-username/<str:username>/', get_user_info, name='get_user_info'),
    path('set-csrf-cookie/', set_csrf_cookie, name='set_csrf_cookie'),
]
