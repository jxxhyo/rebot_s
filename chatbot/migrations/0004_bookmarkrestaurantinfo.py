# Generated by Django 5.0.6 on 2024-06-07 01:28

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("chatbot", "0003_remove_resimage_image_en_url_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="BookmarkRestaurantInfo",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name_ko", models.CharField(max_length=255)),
                ("name_en", models.CharField(max_length=255)),
                ("name_ja", models.CharField(max_length=255)),
                ("name_zh", models.CharField(max_length=255)),
                ("image_ko", models.CharField(blank=True, max_length=255, null=True)),
                ("image_en", models.CharField(blank=True, max_length=255, null=True)),
                ("image_ja", models.CharField(blank=True, max_length=255, null=True)),
                ("image_zh", models.CharField(blank=True, max_length=255, null=True)),
                ("bookmark_count", models.IntegerField(default=0)),
                ("latitude", models.FloatField()),
                ("longitude", models.FloatField()),
            ],
        ),
    ]
