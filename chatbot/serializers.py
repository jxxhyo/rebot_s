# serializers.py
from rest_framework import serializers
from django.contrib.auth.models import User

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
        user.set_password1(validated_data['password1'])
        user.profile.language = validated_data['language']
        user.save()
        return user
