from django.contrib import admin
from .models import Chat, Profile, Restaurant, SavedRestaurant

# Register your models here.
admin.site.register(Chat)
admin.site.register(Profile)

@admin.register(Restaurant)
class RestaurantAdmin(admin.ModelAdmin):
    list_display = ('name', 'category', 'location', 'service', 'menu1', 'menu2')

@admin.register(SavedRestaurant)
class SavedRestaurantAdmin(admin.ModelAdmin):
    list_display = ('user', 'restaurant')
