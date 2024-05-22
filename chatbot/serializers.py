# serializers.py
from rest_framework import serializers
from django.contrib.auth.models import User

class UserSignUpSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    password_confirm = serializers.CharField(write_only=True)
    language = serializers.ChoiceField(choices=[('en', 'English'), ('ko', 'Korean')])

    class Meta:
        model = User
        fields = ['username', 'password', 'password_confirm', 'language']

    def validate(self, data):
        if data['password'] != data['password_confirm']:
            raise serializers.ValidationError("Passwords do not match.")
        return data

    def create(self, validated_data):
        user = User(
            username=validated_data['username'],
        )
        user.set_password(validated_data['password'])
        user.profile.language = validated_data['language']
        user.save()
        return user
