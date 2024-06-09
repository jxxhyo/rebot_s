from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Restaurant, SavedRestaurant, ResImage

class UserSignUpSerializer(serializers.ModelSerializer):
    password1 = serializers.CharField(write_only=True)
    password2 = serializers.CharField(write_only=True)
    language = serializers.ChoiceField(choices=[('en', 'English'), ('ko', 'Korean')])

    class Meta:
        model = User
        fields = ['username', 'password1', 'password2', 'language']

    def validate(self, data):
        if data['password1'] != data['password2']:
            raise serializers.ValidationError("Passwords do not match.")
        return data

    def create(self, validated_data):
        user = User(
            username=validated_data['username'],
        )
        user.set_password(validated_data['password1'])
        user.profile.language = validated_data['language']
        user.save()
        return user

class ResImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResImage
        fields = ['image_name', 'image_en', 'image_ko', 'image_zh', 'image_ja']

class RestaurantSerializer(serializers.ModelSerializer):
    resimages = ResImageSerializer(many=True, read_only=True)

    class Meta:
        model = Restaurant
        fields = '__all__'

class SavedRestaurantSerializer(serializers.ModelSerializer):
    restaurant_name = serializers.CharField(source='restaurant.name', read_only=True)
    restaurant_category = serializers.CharField(source='restaurant.category', read_only=True)
    restaurant_location = serializers.CharField(source='restaurant.location', read_only=True)
    restaurant_images = ResImageSerializer(source='restaurant.resimages', many=True, read_only=True)
    
    class Meta:
        model = SavedRestaurant
        fields = ['id', 'user', 'restaurant', 'restaurant_name', 'restaurant_category', 'restaurant_location', 'restaurant_images']

