from app import app
from flask import jsonify
from app.data_processing import save_orders_to_json, count_products_by_department, get_total_orders_and_users, calculate_reorder_ratios, calculate_reorder_ratio_by_order

import json


@app.route('/')
def home():
    return 'Welcome to my Flask API!'

@app.route('/api/hello', methods=['GET'])
def hello_api():
    return jsonify({'message': 'Hello from the API!'})

# Contoh rute untuk API
@app.route('/api/orders', methods=['GET'])
def get_orders_api():
    # Panggil fungsi dari data_processing.py
    result = save_orders_to_json()

    # Kembalikan hasil sebagai JSON
    return jsonify({'orders': json.loads(result)})

@app.route('/api/departemen', methods=['GET'])
def get_count_products_by_department():
    # Panggil fungsi dari data_processing.py
    result = count_products_by_department()

    # Kembalikan hasil sebagai JSON
    return jsonify({'data': json.loads(result)})

@app.route('/api/total_orders_and_users', methods=['GET'])
def get_total_orders_and_users_api():
    result = get_total_orders_and_users()
    return jsonify({'data': result})

@app.route('/api/rasio_pesanan', methods=['GET'])
def calculate_reorder_ratios_api():
    result = calculate_reorder_ratios()
    return jsonify({'data': result})

@app.route('/api/rasio_by_orders', methods=['GET'])
def calculate_reorder_ratio_by_order_api():
    result = calculate_reorder_ratio_by_order()
    return jsonify({'data': json.loads(result)})