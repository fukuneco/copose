from django.urls import path, include
from .views import PredView


urlpatterns = [

    path('', PredView.as_view(), name='index'),

]