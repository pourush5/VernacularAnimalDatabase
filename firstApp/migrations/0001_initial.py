# Generated by Django 3.0.6 on 2024-05-26 13:30

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ImagePrediction',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('vernacular_name', models.CharField(max_length=255)),
                ('predicted_label', models.CharField(max_length=255)),
                ('image_path', models.CharField(max_length=255)),
            ],
        ),
    ]
