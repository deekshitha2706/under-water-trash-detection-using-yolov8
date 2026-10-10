from django import forms
from .models import DetectionImage, DetectionVideo

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = DetectionImage
        fields = ['image']
        widgets = {
            'image': forms.FileInput(attrs={'class': 'form-control'})
        }

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = DetectionVideo
        fields = ['video']
        widgets = {
            'video': forms.FileInput(attrs={'class': 'form-control'})
        }