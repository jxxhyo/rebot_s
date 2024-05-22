from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)
    language = forms.ChoiceField(choices=[
        ('en', 'English'),
        ('ko', 'Korean'),
        ('zh', 'Chinese'),
        ('ja', 'Japanese')
    ])

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2', 'language']

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
            user.profile.language = self.cleaned_data['language']
            user.profile.save()
        return user
