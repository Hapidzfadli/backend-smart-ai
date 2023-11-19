from flask import Flask
from flask_cors import CORS
# Membuat objek Flask
app = Flask(__name__)
CORS(app)  # Menambahkan header CORS secara otomatis untuk semua rute
# Import modul routes agar rute terdaftar
from app import routes
