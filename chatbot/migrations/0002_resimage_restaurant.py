# Generated by Django 5.0.6 on 2024-06-06 01:14

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("chatbot", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="resimage",
            name="restaurant",
            field=models.ForeignKey(
                default=1,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="resimages",
                to="chatbot.restaurant",
            ),
            preserve_default=False,
        ),
    ]
