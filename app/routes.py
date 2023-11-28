from app import app
from flask import jsonify, request
from app.data_processing import save_orders_to_json,  count_products_by_department, get_total_orders_and_users, calculate_reorder_ratios, calculate_reorder_ratio_by_order, reordered_products_histogram, days_since_prior_order_histogram, order_hour_of_day_histogram, order_day_of_week_histogram, percentage_of_ordering, count_of_ordering, organic_purchase_frequency, organic_ratio, department_order_percentage
from app.apriori import load_and_process_data, select_rules_with_antecedents_names
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

@app.route('/api/reordered_products_histogram', methods=['GET'])
def reordered_products_histogram_api():
    result = reordered_products_histogram()
    return jsonify({'data': result})

@app.route('/api/order_day_of_week_histogram', methods=['GET'])
def order_day_of_week_histogram_api():
    result = order_day_of_week_histogram()
    return jsonify({'data': result})

@app.route('/api/order_hour_of_day_histogram', methods=['GET'])
def order_hour_of_day_histogram_api():
    result = order_hour_of_day_histogram()
    return jsonify({'data': result})

@app.route('/api/percentage_of_ordering', methods=['GET'])
def percentage_of_ordering_api():
    result = percentage_of_ordering()
    return jsonify({'data': json.loads(result)})

@app.route('/api/days_since_prior_order_histogram', methods=['GET'])
def days_since_prior_order_histogram_api():
    result = days_since_prior_order_histogram()
    return jsonify({'data': result})


@app.route('/api/count_of_ordering', methods=['GET'])
def count_of_ordering_api():
    result = count_of_ordering()
    return jsonify({'data': json.loads(result)})


@app.route('/api/organic_ratio', methods=['GET'])
def organic_ratio_api():
    result = organic_ratio()
    return jsonify({'data': result})

@app.route('/api/organic_purchase_frequency', methods=['GET'])
def organic_purchase_frequency_api():
    result = organic_purchase_frequency()
    return jsonify({'data': result})


@app.route('/api/department_order_percentage', methods=['GET'])
def department_order_percentage_api():
    result = department_order_percentage()
    return jsonify({'data': result})


# ASSOSIASI APRIORI
frequent_items, rules = load_and_process_data()
# API route to get recommendations
@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    categories = request.args.getlist('category')
    support_threshold = float(request.args.get('support_threshold', 0.001))
    lift_threshold = float(request.args.get('lift_threshold', 1.1))

    if not categories:
            return jsonify({'error': 'At least one category is required'})
    # Filter frequent_items based on the category
    # filtered_items = frequent_items[frequent_items['itemsets'].apply(lambda x: category in x)]

    # Filter rules based on the support and lift thresholds
    filtered_rules = rules[
        (rules['antecedent_len'] >= 1) &
        (rules['confidence'] >= 0.3) &
        (rules['lift'] >= lift_threshold)
    ]
    
    category_set = set(categories)

    product = select_rules_with_antecedents_names(rules, category_set)
    consequents_column = product['antecedents']
    # Convert the frozensets to lists
    consequents_list_of_lists = consequents_column.apply(lambda x: list(x))
    # Convert the lists to JSON
    json_result = consequents_list_of_lists.to_json(orient='records')
    # Print or use the JSON result as needed

    return json_result
    

