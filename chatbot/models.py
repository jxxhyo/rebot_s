from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

class Chat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username}: {self.message}'

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    language = models.CharField(max_length=2, choices=[
        ('en', 'English'),
        ('ko', 'Korean'),
        ('zh', 'Chinese'),
        ('ja', 'Japanese')
    ])

# 신호를 사용하여 프로필 자동 생성
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()

# User 모델에 profile 속성을 추가
def get_or_create_profile(self):
    profile, created = Profile.objects.get_or_create(user=self)
    return profile

User.add_to_class("profile", property(get_or_create_profile))

class Restaurant(models.Model):
    name = models.CharField(max_length=255)
    category = models.CharField(max_length=255)
    location = models.CharField(max_length=255)
    service = models.CharField(max_length=255, blank=True, null=True)
    menu1 = models.CharField(max_length=255, blank=True, null=True)
    menu2 = models.CharField(max_length=255, blank=True, null=True)
    mood = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return self.name

class SavedRestaurant(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    restaurant = models.ForeignKey(Restaurant, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.user.username} saved {self.restaurant.name}"

class ResImage(models.Model):
    restaurant = models.ForeignKey(Restaurant, related_name='resimages', on_delete=models.CASCADE)
    image_name = models.CharField(max_length=255)
    image_en = models.ImageField(upload_to='images/en/', blank=True, null=True)
    image_ko = models.ImageField(upload_to='images/ko/', blank=True, null=True)
    image_zh = models.ImageField(upload_to='images/zh/', blank=True, null=True)
    image_ja = models.ImageField(upload_to='images/ja/', blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.image_name


class BookmarkRestaurantInfo(models.Model):
    name_ko = models.CharField(max_length=255)
    name_en = models.CharField(max_length=255)
    name_ja = models.CharField(max_length=255)
    name_zh = models.CharField(max_length=255)
    image_ko = models.CharField(max_length=255, blank=True, null=True)
    image_en = models.CharField(max_length=255, blank=True, null=True)
    image_ja = models.CharField(max_length=255, blank=True, null=True)
    image_zh = models.CharField(max_length=255, blank=True, null=True)
    bookmark_count = models.IntegerField(default=0)
    latitude = models.FloatField()
    longitude = models.FloatField()


    def __str__(self):
        return self.name_ko
