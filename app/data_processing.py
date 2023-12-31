import numpy as np
import pandas as pd
import os
import zipfile
import glob
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
from IPython.display import display
import random


data_directory_path = 'csv/'
# Mengonversi hari dan jam dari angka ke bentuk yang dapat dimengerti
days_of_week = {0: 'Sabtu',
                1: 'Minggu',
                2: 'Senin',
                3: 'Selasa',
                4: 'Rabu',
                5: 'Kamis',
                6: 'Jumat'}
hour_nums = list(range(24))
hours_of_day = {hour_num: f"{hour_num:02}.00" for hour_num in hour_nums}

'''
    Iterasi melalui semua kolom dalam dataframe dan mengubah tipe data
    untuk mengurangi penggunaan memori.
'''
def reduce_mem_usage(train_data):
    start_mem = train_data.memory_usage().sum() / 1024**2
    print('Penggunaan memori dari dataframe adalah {:.2f} MB'.format(start_mem))

    for col in train_data.columns:
        col_type = train_data[col].dtype

        if col_type not in [object, 'category']:
            c_min = train_data[col].min()
            c_max = train_data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    train_data[col] = train_data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    train_data[col] = train_data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    train_data[col] = train_data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    train_data[col] = train_data[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    train_data[col] = train_data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    train_data[col] = train_data[col].astype(np.float32)
                else:
                    train_data[col] = train_data[col].astype(np.float64)
        else:
            train_data[col] = train_data[col].astype('category')
    end_mem = train_data.memory_usage().sum() / 1024**2
    print('Penggunaan memori setelah optimalisasi adalah: {:.2f} MB'.format(end_mem))
    print('Berkurang sebesar {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return train_data

# Membaca file csv ke dalam dataframe yang sesuai
# Kemudian mengurangi ukuran mereka untuk mengonsumsi memori lebih sedikit
aisles = pd.read_csv(data_directory_path + 'aisles.csv')
aisles = reduce_mem_usage(aisles)

departments = pd.read_csv(data_directory_path + 'departments.csv')
departments = reduce_mem_usage(departments)

order_products_prior = pd.read_csv(data_directory_path + 'order_products__prior.csv')
order_products_prior = reduce_mem_usage(order_products_prior)

order_products_train = pd.read_csv(data_directory_path + 'order_products__train.csv')
order_products_train = reduce_mem_usage(order_products_train)

orders = pd.read_csv(data_directory_path + 'orders.csv')

# Mengganti angka dalam kolom 'order_hour_of_day' dan 'order_dow'
orders['order_hour_of_day'] = orders['order_hour_of_day'].map(hours_of_day)
orders['order_dow'] = orders['order_dow'].map(days_of_week)

# Mengoptimalkan penggunaan memori untuk DataFrame 'orders'
orders = reduce_mem_usage(orders)

products = pd.read_csv(data_directory_path + 'products.csv')
products = reduce_mem_usage(products)

order_products = pd.concat([order_products_train, order_products_prior])
order_products = order_products.merge(products, on='product_id', how='left').merge(orders, on='order_id', how='left').merge(departments, on='department_id').merge(aisles, on='aisle_id')

# Menghitung total pesanan
total_orders = order_products['order_id'].nunique()
total_users = order_products['user_id'].nunique()


def save_orders_to_json():
    # Pengolahan data (seperti yang telah dijelaskan sebelumnya)

    # Menggabungkan data dari beberapa DataFrames (misalnya, orders, order_products_train, dsb.)
    merged_data = pd.merge(orders, order_products_train, on='order_id', how='inner')

    # Pilih kolom-kolom yang ingin Anda sertakan dalam hasil JSON
    selected_columns = ['order_id', 'user_id', 'product_id']

    # Buat DataFrame baru hanya dengan kolom yang dipilih
    selected_data = merged_data[selected_columns].head(10)

    # Ubah DataFrame menjadi format JSON
    json_result = selected_data.to_json(orient='records')

    return json_result

def count_orders_by_eval_set():
    # Menggunakan data dari DataFrame 'orders'
    order_counts = orders['eval_set'].value_counts().reset_index()
    order_counts.columns = ['eval_set', 'count']
 
    # Ubah DataFrame menjadi format JSON
    json_result = order_counts.to_json(orient='records')

    return json_result


def count_products_by_department():
    # Gabungkan produk dengan departemen
    prod_dept = products.merge(departments, on='department_id', how='left')

    # Hitung jumlah produk dalam setiap departemen
    data = prod_dept.groupby(['department']).agg({'product_id':'count'}).reset_index().rename(columns={'product_id':'products_count'})

    # Ubah DataFrame menjadi format JSON
    json_result = data.to_json(orient='records')

    return json_result

def calculate_reorder_ratio_by_order():
    # Hitung rasio pemesanan ulang untuk setiap nomor pesanan
    groupeddf = order_products.groupby(['add_to_cart_order']).agg({'reordered':'mean'}).rename(columns={'reordered':'reordered_ratio'}).reset_index()
    
    # Ubah DataFrame menjadi format JSON
    json_result = groupeddf.to_json(orient='records')

    return json_result


def get_total_orders_and_users():
    result = {
        'total_orders': total_orders,
        'total_users': total_users
    }

    return result

def calculate_reorder_ratios():
    # Mengelompokkan data berdasarkan 'order_id' dan menghitung rerata dari kolom 'reordered'
    groupeddf = order_products.groupby(['order_id']).agg({'reordered': 'mean', 'order_number': 'first'}).reset_index().rename(columns={'reordered': 'reordered_ratio'})
    groupeddf['no_reordered'] = groupeddf['reordered_ratio'] == 0
    groupeddf['all_reordered'] = groupeddf['reordered_ratio'] == 1.0

    # Menghitung jumlah pesanan yang bukan pesanan pertama
    non_first_orders_Mask = orders.order_number != 1
    non_first_orders_count = np.sum(non_first_orders_Mask)

    # Batasi jumlah data yang diolah untuk mengoptimalkan performa
    groupeddf = groupeddf.sample(frac=0.1, random_state=42)

    # Menghitung rasio pesanan yang bukan pesanan pertama yang tidak mengandung produk yang pernah dipesan sebelumnya vs. mengandung setidaknya satu produk yang pernah dipesan sebelumnya
    groupeddf = groupeddf[groupeddf.order_number != 1]
    groupeddf['no_reordered'] = groupeddf['reordered_ratio'] == 0
    no_reordered_count = groupeddf['no_reordered'].value_counts() / non_first_orders_count * 100
    
    # Rasio pesanan yang bukan pesanan pertama di mana semua produknya pernah dibeli sebelumnya
    groupeddf['all_reordered'] = groupeddf['reordered_ratio'] == 1.0
    all_reordered_count = groupeddf['all_reordered'].value_counts()/non_first_orders_count * 100

    # Persentase pesanan yang tidak mengandung produk yang pernah dipesan sebelumnya
    rasio = {
        'no_reordered': no_reordered_count.values.tolist(),
        'all_reordered': all_reordered_count.values.tolist()
    }

    # Menggabungkan hasil dalam satu objek
    result = rasio

    return result

def reordered_products_histogram():
    # Membuat histogram untuk jumlah produk yang dipesan ulang dalam satu pesanan
    count_of_reordered_products = order_products.groupby(['order_id'], as_index=False)['reordered'].sum()
    count_of_reordered_products['count_of_reordered_products'] = count_of_reordered_products['reordered']
    less_than_50_reordered_products = count_of_reordered_products[count_of_reordered_products['count_of_reordered_products'] < 50]

    # Membuat histogram dan menyimpan data ke dalam format yang diinginkan
    histogram_data, bin_edges = np.histogram(less_than_50_reordered_products['count_of_reordered_products'], bins=100)
    histogram_result = [{'bin': int(bin_edge), 'count': int(count)} for bin_edge, count in zip(bin_edges, histogram_data) if count != 0]

    # Hasil yang diinginkan
    result = histogram_result

    return result


def order_day_of_week_histogram():
    # Count occurrences of each day in 'order_dow'
    day_counts = orders['order_dow'].value_counts().sort_index()

    # Convert the result to a list of dictionaries
    histogram_result = [{'day': day, 'count': count} for day, count in day_counts.items()]

    # Result as a dictionary
    result = histogram_result
    return result


def order_hour_of_day_histogram():
    # Membuat histogram untuk jam dalam sehari
    histogram_data, bin_edges = np.histogram(orders['order_hour_of_day'], bins=24)
    histogram_result = [{'bin': int(bin_edge), 'count': int(count)} for bin_edge, count in zip(bin_edges, histogram_data)]

    # Hasil yang diinginkan
    result = {'histogram_data': histogram_result}
    return result

def days_since_prior_order_histogram():
    # Drop rows with NaN values in 'days_since_prior_order'
    orders_no_nan = orders.dropna(subset=['days_since_prior_order'])

    # Membuat histogram untuk jumlah hari sebelum pesanan terakhir
    histogram_data, bin_edges = np.histogram(orders_no_nan['days_since_prior_order'], bins=30)
    histogram_result = [{'bin': int(bin_edge), 'count': int(count)} for bin_edge, count in zip(bin_edges, histogram_data)]

    # Hasil yang diinginkan
    result = histogram_result
    return result


def percentage_of_ordering():
    # Menyortir produk secara menurun berdasarkan frekuensi pemesanannya
    products_frequency_desc = order_products.groupby('product_name')['order_id'].aggregate('count').reset_index()
    products_frequency_desc['count_of_ordering'] = products_frequency_desc['order_id']
    products_frequency_desc = products_frequency_desc.sort_values(by=['count_of_ordering'], ascending=False)

    # Mengambil 30 produk yang paling sering dipesan
    top_products = products_frequency_desc.iloc[:30]

    # Persentase Pemesanan untuk 30 Produk yang Paling Sering Dipesan
    percentage_data = top_products[['product_name', 'count_of_ordering']].copy()
    percentage_data['percentage_of_ordering'] = (percentage_data['count_of_ordering'] / total_orders) * 100

    # Persentase Pemesanan JSON
    percentage_json = percentage_data.to_json(orient='records')

    return percentage_json


def count_of_ordering():
    # Menyortir produk secara menurun berdasarkan frekuensi pemesanannya
    products_frequency_desc = order_products.groupby('product_name')['order_id'].aggregate('count').reset_index()
    products_frequency_desc['count_of_ordering'] = products_frequency_desc['order_id']
    products_frequency_desc = products_frequency_desc.sort_values(by=['count_of_ordering'], ascending=False)

    # Mengambil 30 produk yang paling sering dipesan
    top_products = products_frequency_desc.iloc[:30]

    # Jumlah Pesanan JSON
    count_json = top_products[['product_name', 'count_of_ordering']].to_json(orient='records')

    return count_json

def is_organic(product_name):
    return 'Organic' in product_name



def organic_ratio():
    # Rasio produk organik yang ditemukan di supermarket
    products['is_organic'] = products['product_name'].apply(is_organic)
    total_products_count = products.shape[0]
    organic_ratio = products['is_organic'].value_counts() / total_products_count * 100

    # Convert the result to JSON
    result_json = organic_ratio.to_dict()

    return result_json

def organic_purchase_frequency():
    # Frekuensi pembelian produk organik
    products['is_organic'] = products['product_name'].apply(is_organic)
    total_order_products_count = order_products.shape[0]
    merged_df = order_products.merge(products, on='product_id', how='inner')
    organic_freq = merged_df['is_organic'].value_counts() / merged_df.shape[0] * 100

    # Convert the result to JSON
    result_json = {
        'organic_purchase_frequency': organic_freq.to_dict(),
    }

    return result_json


def department_order_percentage():
    # Dari Seluruh Produk yang Dipesan, Jumlah Kontribusi dari Setiap Departemen
    # Mengelompokkan berdasarkan jumlah pemesanan dari suatu departemen
    total_ordered_products = order_products.shape[0]
    groupeddf = order_products.groupby(['department'], as_index=False).count().rename(columns={'department_id': 'count_of_ordered_prods'})

    # Calculate percentage and create a list of dictionaries
    result_list = [
        {'value': percentage, 'label': department.capitalize(), 'id': department.lower()}
        for department, percentage in zip(groupeddf['department'], groupeddf['count_of_ordered_prods'] / total_ordered_products * 100)
    ]

    # Convert the result to JSON
    result_json = result_list

    return result_json

