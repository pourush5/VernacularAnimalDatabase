from django.db import models

class ImagePrediction(models.Model):
    vernacular_name = models.CharField(max_length=255)
    language = models.CharField(max_length=255)  # Add this field
    predicted_label = models.CharField(max_length=255)
    image_path = models.CharField(max_length=255)

    def __str__(self):
        return self.vernacular_name
