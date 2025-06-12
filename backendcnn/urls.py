from django.contrib import admin
from django.urls import path, re_path
from backendcnn.models import views

urlpatterns = [
    path('admin/', admin.site.urls),
    re_path(r'^model/?$', views.hello, name='hello'),
    re_path(r'^reset/?$', views.reset, name='reset'),
    re_path(r'^predict/?$', views.predict, name='predict')
]