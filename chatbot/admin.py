from django.contrib import admin
from .models import Chat, Profile, Restaurant, SavedRestaurant, ResImage

# Register your models here.
admin.site.register(Chat)
admin.site.register(Profile)

@admin.register(Restaurant)
class RestaurantAdmin(admin.ModelAdmin):
    list_display = ('name', 'category', 'location', 'service', 'menu1', 'menu2')

@admin.register(SavedRestaurant)
class SavedRestaurantAdmin(admin.ModelAdmin):
    list_display = ('user', 'restaurant')

@admin.register(ResImage)
class ResImageAdmin(admin.ModelAdmin):
    list_display = ('image_name', 'restaurant', 'image_en', 'image_ko', 'image_zh', 'image_ja', 'uploaded_at')
