"""
Django settings for emotion_monitor project.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("SECRET_KEY", default="emotion-monitor-secret-key-dev")

DEBUG = os.environ.get("DEBUG", "True").lower() in ["true", "yes", "1"]

ALLOWED_HOSTS = ["localhost", "0.0.0.0", "127.0.0.1"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "emotion_monitor.apps.EmotionMonitorConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [
    BASE_DIR / "static",
]

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Emotion Monitor Settings
FACE2NODES_MODEL_PATH = r"D:\Users\cyz\dc\moxing\face2nodes_best.pth"
VISUAL_EMOTION_MODEL_PATH = BASE_DIR / "ml_models" / "visual_emotion_best.pth"
VISUAL_FEATURE_DIR = r"D:\Users\cyz\dc\see"
DEAP_MAT_DIR = r"E:\BaiduNetdiskDownload\DEAP\data_preprocessed_matlab"
CAMERA_INDEX = 0
VISUAL_TARGET_FEATURES = 15
VISUAL_OUTPUT_DIM = 512

MHDGFNET_MODEL_PATH = r"D:\Users\cyz\dc\moxing\mh_dgfnet_s01_s22_best.pth"
MHDGFNET_SCALER_PATH = r"D:\Users\cyz\dc\moxing\scaler_s01_s22_best.pth"
MHDGFNET_SEGMENTS_PER_TRIAL = 15
MHDGFNET_KEEP_LAST = 15
MULTIMODAL_MODEL_PATH = MHDGFNET_MODEL_PATH
MULTIMODAL_SCALER_PATH = MHDGFNET_SCALER_PATH
