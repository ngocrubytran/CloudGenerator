from django.http import HttpResponse
from django.shortcuts import render
# for storing the uploaded image
from django.core.files.storage import FileSystemStorage
import os
import numpy as np
from matplotlib.image import imread
from mat4py import loadmat
from PIL import Image

# Create your views here.


def index(request):
    return render(request, 'home.html')


def process(request):
    # remove all the previous uploaded pictures
    if(request.method == "POST"):
        if(request.POST.get('upload-button')):
            dir = "media"
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))
            if request.FILES['upload']:
                upload = request.FILES['upload']
                fss = FileSystemStorage()
                my_image = fss.save(upload.name, upload)
                image_url = fss.url(my_image)
                return render(request, 'home.html', {'image_url': image_url})
        elif(request.POST.get('process-button')):
            image_url = request.POST.get('image-url')
            if(image_url):
                slider_value = float(request.POST.get('slider', None))

                # we already have the url to the image, now we need to process the image and post it
                masked_image, width, height = masking(image_url)
                transformed_image = cloudIdentification(
                    masked_image, width, height, slider_value)
                return render(request, 'home.html', {'slider_value': slider_value, 'image_url': image_url, 'transformed_image': transformed_image})
            else:
                render(request, 'home.html')
    return render(request, 'home.html')


def download(request):
    pass


def masking(filename: str) -> None:
    filename = "."+filename
    org_img = imread(filename)

    # load mask image and convert to numpy array
    mask = loadmat("CloudGenerator/static/mask.mat")
    mask_arr = np.array(mask["mask"])

    segment_img = org_img.copy()
    width, height = segment_img.shape[0], segment_img.shape[1]

    # for all values in mask_arr, if it equals 0, set RGB values at that index in segment_img to 0
    for idx, val in np.ndenumerate(mask_arr):
        if val == 0:
            for i in range(3):
                segment_img[idx][i] = 0

    return segment_img, width, height


def cloudIdentification(segment_img: Image, width: int, height: int, slider_value: int):
    width, height = segment_img.shape[0], segment_img.shape[1]
    data = np.array(segment_img).reshape((width*height), 3)
    data = data[:, 0] / (data[:, 2])  # R/B selection criterion
    # for all values in data, if it is not a number set it equal to 0
    data[np.isnan(data)] = 0
    # for all values in data, if it is infinity set it equal to 0
    data[np.isinf(data)] = 0

    # reshape data
    data = np.reshape(data, (width, height))

    for w in range(width):
        for h in range(height):
            # print(slider_value)
            if data[w][h] < slider_value:
                for i in range(3):
                    segment_img[w][h][i] = 0
                    # print(segment_img[w][h][i])
            else:
                # print(data[w][h])
                for i in range(3):
                    segment_img[w][h][i] = 255
                    # print(segment_img[w][h][i])

    transformed_image = "CloudGenerator/static/output.png"
    im = Image.fromarray(segment_img)
    im.save(transformed_image)

    return transformed_image
