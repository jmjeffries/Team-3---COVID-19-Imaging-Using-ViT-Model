from django.db import models

class UploadImage(models.Model):  
    name = models.CharField(max_length=200)  
    img = models.ImageField(upload_to='uploaded/')  
  
    def __str__(self):  
        return self.name  
