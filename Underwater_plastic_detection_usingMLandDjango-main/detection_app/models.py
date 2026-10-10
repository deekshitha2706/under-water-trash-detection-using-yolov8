from django.db import models
class DetectionImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    result_image = models.ImageField(upload_to='results/', blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Image {self.id} - {self.uploaded_at}"

class DetectionVideo(models.Model):
    video = models.FileField(upload_to='uploads/')
    result_video = models.FileField(upload_to='results/', blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Video {self.id} - {self.uploaded_at}"
# Create your models here.
