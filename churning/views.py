import os
from django.shortcuts import render
from django.http import HttpResponse, Http404
from .forms import CustomerForm
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from . import AnnProcessing

# Create your views here.


def dataview(request):
    form = CustomerForm()
    if request.method == "POST":
        form = CustomerForm(request.POST, request.FILES)
        if form.is_valid():
            
            customer_file = request.FILES['customer_file']
            fs = FileSystemStorage()
            customer_file = fs.save('CustomerAccounts.csv', customer_file)
            
            #Here is where my code goes 
            
            directorypath = settings.MEDIA_ROOT
            AnnProcessing.churningPrediction(directorypath)
            
            
            return render(request, 'final.html')
    
    
    return render(request, 'dataview.html', {'form' : form })

    


def download(request):
    file_path = os.path.join(settings.MEDIA_ROOT, 'Customer_file_with_predictions.csv')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response
    raise Http404
