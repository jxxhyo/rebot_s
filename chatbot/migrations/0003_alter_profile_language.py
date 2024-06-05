# Generated by Django 5.0.6 on 2024-06-04 06:29

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("chatbot", "0002_delete_customuser"),
    ]

    operations = [
        migrations.AlterField(
            model_name="profile",
            name="language",
            field=models.CharField(
                choices=[
                    ("en", "English"),
                    ("ko", "Korean"),
                    ("zh", "Chinese"),
                    ("ja", "Japanese"),
                ],
                max_length=2,
            ),
        ),
    ]