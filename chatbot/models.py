from django.db import models
from django.contrib.auth.models import User
from django.db import models
from django.contrib.auth.models import AbstractUser

# Create your models here.
class Chat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username}: {self.message}'
    
from django.db import models
from django.contrib.auth.models import AbstractUser

class CustomUser(AbstractUser):
    LANGUAGE_CHOICES = [
        ('en', 'English'),
        ('ko', 'Korean'),
        ('zh', 'Chinese'),
        ('ja', 'Japanese'),
    ]
    language = models.CharField(max_length=2, choices=LANGUAGE_CHOICES, default='en')

    groups = models.ManyToManyField(
        'auth.Group',
        related_name='customuser_set',  # related_name 속성을 설정하여 충돌을 피합니다
        blank=True,
        help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.',
        verbose_name='groups'
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='customuser_set',  # related_name 속성을 설정하여 충돌을 피합니다
        blank=True,
        help_text='Specific permissions for this user.',
        verbose_name='user permissions'
    )
