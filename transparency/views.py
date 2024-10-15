from django.shortcuts import render, redirect
from django.conf import settings
from .forms import ImageUploadForm 
from .models import UploadImage



#render pages
def index(request):
    return render(request, 'index.html')

def login(request):
    return render(request, 'login.html')

def result(request):
    return render(request, 'result.html')

def upload_image(request): 
    print('here')
    if request.method == 'POST':  
        form = ImageUploadForm(request.POST, request.FILES)  # Correct the form class reference
        if form.is_valid():  
            form.save()  
            # Getting the current instance object to display in the template  
            img_object = form.instance  
            # Redirect to the same page after successful submission to prevent form resubmission
            return render(request, 'index.html', {'form': form, 'img_obj': img_object})  
    else:  
        form = ImageUploadForm()  
  
    return render(request, 'index.html', {'form': form})  

# def analyze_image(img_path):
#     # Load and preprocess the image
#     img = image.load_img(img_path, target_size=(72, 72))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0

#     # Predict using the model
#     predictions = model.predict(img_array)
#     return predictions


