from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    register, 
    login_view, 
    logout_view, 
    chatbot, 
    get_user_info, 
    set_csrf_token,  # set_csrf_cookie 뷰 추가
    RestaurantViewSet, 
    saved_restaurants, 
    recommend_restaurants, 
    save_restaurant, 
    unsave_restaurant,
    get_restaurant_coordinates,
    chat_history
)

from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

router = DefaultRouter()
router.register(r'restaurants', RestaurantViewSet)

urlpatterns = [
    path('register/', register, name='register'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('chatbot/', chatbot, name='chatbot'),
    path('get-username/<str:username>/', get_user_info, name='get_user_info'),
    path('set-csrf-token/', set_csrf_token, name='set_csrf_token'),  # URL 패턴 추가
    path('api/saved-restaurants/', saved_restaurants, name='saved_restaurants'),
    path('api/recommend-restaurants/', recommend_restaurants, name='recommend_restaurants'),
    path('api/save-restaurant/', save_restaurant, name='save_restaurant'),
    path('api/unsave-restaurant/', unsave_restaurant, name='unsave_restaurant'),
    path('api/', include(router.urls)),  # Include the router URLs
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('get-restaurant-coordinates/', get_restaurant_coordinates, name='get_restaurant_coordinates'),
    path('chat_history/', chat_history, name='chat_history'),
]
    #path('api-auth/', include('rest_framework.urls')),  # DRF 기본 인증 URL 포함


from django.conf import settings
from django.conf.urls.static import static

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)