from django.db import models  
from django.forms import fields  
from django import forms
from .models import UploadImage

# class ImageUploadForm(forms.ModelForm):
#     class Meta:
#         models = UploadImage
#         fields = [
#             'name'
#             'image',
#         ]

class ImageUploadForm(forms.Form):
    name = forms.CharField(max_length=50)
    image = forms.ImageField()