from django.urls import path
from . import views

app_name = "emotion_monitor"

urlpatterns = [
    path("", views.EmotionMonitorView.as_view(), name="page"),


    path("api/model-status/", views.model_status, name="model_status"),

    path(
        "api/predict-deap-separated-upload/",
        views.predict_deap_separated_upload,
        name="predict_deap_separated_upload"
    ),

    path(
    "api/predict-trial-separated/",
    views.predict_trial_separated,
    name="predict_trial_separated"
    ),

    path("api/dashboard-data/", views.dashboard_data, name="dashboard_data"),
]
