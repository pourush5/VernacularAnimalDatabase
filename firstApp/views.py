from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import json
from tensorflow import Graph
from tensorflow.compat.v1 import Session
import numpy as np
from .models import ImagePrediction
from django.db.models import Q


img_height, img_width = 224, 224

with open('./models/imagenet_classes.json', 'r') as f:
    labelInfo = json.loads(f.read())

model_graph = Graph()
with model_graph.as_default():
    tf_session = Session()
    with tf_session.as_default():
        model = load_model('./models/MobileNetModelImagenet.h5')

def index(request):
    context = {'a': 1}
    return render(request, 'index.html', context)

def predictImage(request):
    if request.method == 'POST':
        fileObj = request.FILES['filePath']
        vernacularName = request.POST['vernacularName']
        language = request.POST['language']
        fs = FileSystemStorage()
        filePathName = fs.save(fileObj.name, fileObj)
        filePathName = fs.url(filePathName)
        testimage = '.' + filePathName
        img = image.load_img(testimage, target_size=(img_height, img_width))
        x = image.img_to_array(img)
        x = x / 255
        x = x.reshape(1, img_height, img_width, 3)
        with model_graph.as_default():
            with tf_session.as_default():
                predi = model.predict(x)
        predictedLabel = labelInfo[str(np.argmax(predi[0]))]

        new_entry = ImagePrediction(vernacular_name=vernacularName, language=language, predicted_label=predictedLabel[1], image_path=filePathName)
        new_entry.save()

        context = {'filePathName': filePathName, 'predictedLabel': predictedLabel[1]}
        return render(request, 'index.html', context)

def viewDataBase(request):
    images = ImagePrediction.objects.all()
    context = {'images': images}
    return render(request, 'viewDB.html', context)

def about(request):
    return render(request,'about.html')

def searchDatabase(request):
    query = request.GET.get('q')
    results = []
    if query:
        results = ImagePrediction.objects.filter(
            Q(vernacular_name__icontains=query) | 
            Q(predicted_label__icontains=query) |
            Q(image_path__icontains=query)
        )
    context = {'results': results, 'query': query}
    return render(request, 'searchDB.html', context)