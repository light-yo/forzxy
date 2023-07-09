"""Django_vue URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from job import views
from django.urls import path
from django.views.generic.base import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('upload1/',views.upload1),
    path('upload2/',views.upload2),
    path('upload3/',views.upload3),
    path('canshu/',views.canshu),
    path('train/',views.start_train),
    path('test/',views.start_test),
    path('plot/',views.get_plot),
    path('plot1/',views.get_plot1),
    path('plot2/',views.get_plot2),
    path('verify/',views.start_verify),
    path('download_model/',views.download_model),
    path('download_test/',views.download_test),
    path('download_verify/',views.download_verify),
    path('', TemplateView.as_view(template_name='index.html'))
]
