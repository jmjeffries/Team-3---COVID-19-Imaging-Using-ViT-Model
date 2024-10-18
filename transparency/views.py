import os
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.conf import settings
from .forms import ImageUploadForm 
from .models import UploadImage
# import keras


#render pages
def home(request):
    return render(request, 'home.html')

#saves image and name combination to folder
def upload_image(request):
    context = {}
    if request.method == 'POST':  
        print(request.FILES)
        form = ImageUploadForm(request.POST, request.FILES) 
        if form.is_valid():  
            name = form.cleaned_data.get("name")
            img = form.cleaned_data.get("image") 

            #changing file name to given name + number of files in the folder
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded')
            num_images = len(os.listdir(upload_dir))
            ext = img.name.split('.')[-1]
            new_filename = f"{name}_{num_images + 1}.{ext}"
            img.name = new_filename


            image_obj = UploadImage.objects.create(
                                        name = name, 
                                        img = img
                                        )
            image_obj.save()
            print(image_obj) 
    else:
        form = ImageUploadForm()
    context['form'] = form
    return render(request, 'home.html', context)



def login(request):
    return render(request, 'login.html')

def result(request):
    return render(request, 'result.html')


# def analyze_image(img_path):
#     # Load and preprocess the image
#     img = image.load_img(img_path, target_size=(72, 72))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0

#     # Predict using the model
#     predictions = model.predict(img_array)
#     return predictions


