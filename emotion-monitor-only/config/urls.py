"""
URL configuration for emotion_monitor project.
"""
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("emotion/", include("emotion_monitor.urls")),
    path("emotion-monitor/", include("emotion_monitor.urls")),
]
