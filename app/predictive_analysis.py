import numpy as np 
import pandas as pd 
import gc
gc.enable()
data_directory_path = 'csv/'

# Membaca file csv ke dalam dataframe yang sesuai
# Kemudian mengurangi ukuran mereka untuk mengonsumsi memori lebih sedikit
aisles = pd.read_csv(data_directory_path + 'aisles.csv')
departments = pd.read_csv(data_directory_path + 'departments.csv')
order_products_prior = pd.read_csv(data_directory_path + 'order_products__prior.csv')
order_products_train = pd.read_csv(data_directory_path + 'order_products__train.csv')
orders = pd.read_csv(data_directory_path + 'orders.csv')
products = pd.read_csv(data_directory_path + 'products.csv')


def reduce_mem_usage(train_data):

    # mengiterasi melalui semua kolom dari dataframe dan mengubah tipe data untuk mengurangi penggunaan memori.
    start_mem = train_data.memory_usage().sum() / 1024**2
    print('Penggunaan memori dari dataframe adalah {:.2f} MB'.format(start_mem))

    for col in train_data.columns:
        col_type = train_data[col].dtype

        if col_type != object:
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
        print('Menurun sebesar {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return train_data

reduce_mem_usage(order_products_prior)
reduce_mem_usage(order_products_train)
reduce_mem_usage(products)
reduce_mem_usage(orders)
reduce_mem_usage(departments)
reduce_mem_usage(aisles)

# Mendapatkan bentuk (shape) dari setiap dataframe
print(f" aisles : {aisles.shape} \n depts : {departments.shape} \n order_prod_prior : {order_products_prior.shape} \n order_products_train : {order_products_train.shape} \n orders : {orders.shape} \n products : {products.shape}")

# Fungsi pembantu untuk menghitung rata-rata waktu
# Permasalahan adalah jika kita berurusan dengan waktu dan menghitung rata-rata mereka secara normal, rata-rata antara 1:00 dan 23:00 akan menjadi 12 bukan 0:00
import datetime
import math

def datetime_to_radians(x):
    # Radian dihitung dengan menggunakan lingkaran 24 jam, bukan 12 jam, dimulai dari utara dan bergerak searah jarum jam
    seconds_from_midnight = 3600 * x
    radians = float(seconds_from_midnight) / float(12 * 60 * 60) * 2.0 * math.pi
    return radians

def average_angle(angles):
    # Sudut diukur dalam radian
    x_sum = np.sum(np.sin(angles))
    y_sum = np.sum(np.cos(angles))
    x_mean = x_sum / float(len(angles))
    y_mean = y_sum / float(len(angles))
    return np.arctan2(x_mean, y_mean)

def radians_to_time_of_day(x):
    # Radian diukur searah jarum jam dari utara dan mewakili waktu dalam lingkaran 24 jam
    seconds_from_midnight = int(float(x) / (2.0 * math.pi) * 12.0 * 60.0 * 60.0)
    hour = seconds_from_midnight // 3600 % 24
    minute = (seconds_from_midnight % 3600) // 60
    second = seconds_from_midnight % 60
    return datetime.time(hour, minute, second)

def average_times_of_day(x):
    # Masukan berupa array datetime.datetime dan keluaran berupa nilai datetime.time
    angles = [datetime_to_radians(y) for y in x]
    avg_angle = average_angle(angles)
    return radians_to_time_of_day(avg_angle)

def day_to_radians(day):
    radians = float(day) / float(7) * 2.0 * math.pi
    return radians
def radians_to_days(x):
    day = int(float(x) / (2.0 * math.pi) * 7) % 7
    return day
def average_days(x):
    angles = [day_to_radians(y) for y in x]
    avg_angle = average_angle(angles)
    return radians_to_days(avg_angle)

# 2.1 FITUR PREDIKSI

# Kami hanya menyimpan pesanan sebelumnya
users = orders[orders['eval_set'] == 'prior']
users['days_since_prior_order'].dropna()

# Kami mengelompokkan pesanan berdasarkan user_id & menghitung variabel berdasarkan user_id yang berbeda
users = users.groupby('user_id').agg(

 user_orders= ('order_number' , max),
 user_period=('days_since_prior_order', sum),
 user_mean_days_since_prior = ('days_since_prior_order','mean')

)
users.head()

# Membuat tabel baru "orders_products" yang berisi tabel "orders" dan "order_products_prior"
orders_products =pd.merge(orders , order_products_prior, on='order_id', how='inner')

# Mengambil jumlah produk dalam setiap keranjang (pesanan)
groupedorders_products = orders_products.groupby(['order_id']).agg(
    basket_size = ('product_id', 'count')
).reset_index()
orders_products = orders_products.merge(groupedorders_products, on='order_id', how='left')
orders_products.head()

orders_products['p_reordered']= orders_products['reordered']==1
orders_products['non_first_order']= orders_products['order_number']>1

us=orders_products

# Mengelompokkan 'orders_products' berdasarkan 'user_id' dan menghitung variabel-variabel berdasarkan 'user_id' yang berbeda
us=orders_products.groupby('user_id').agg(

     user_total_products =('user_id','count') ,
     p_reordered =('p_reordered', sum) ,
     non_first_order =('non_first_order', sum),
     user_distinct_products=('product_id','nunique')

).reset_index()
#    us['user_reorder_ratio'] = sum(reordered == 1) / sum(order_number > 1)
us['user_reorder_ratio']=us['p_reordered']/us['non_first_order']

del us["p_reordered"],us["non_first_order"]
del orders_products['p_reordered' ],orders_products['non_first_order']

us.head(20)

users =pd.merge(users,us ,on='user_id',  how='inner')

# Menghitung variabel 'user_average_basket' sebagai rata-rata jumlah item dalam keranjang per pesanan per pengguna
users['user_average_basket'] = users['user_total_products'] / users['user_orders']
users.head()

# Kami mengeluarkan pesanan sebelumnya dan hanya menyimpan pesanan train dan test
us = orders[orders['eval_set'] != 'prior']
us['time_since_last_order'] = us['days_since_prior_order']
us['future_order_dow'] = us['order_dow']
us['future_order_hour_of_day'] = us['order_hour_of_day']

us = us[['user_id','order_id','eval_set','time_since_last_order', 'future_order_dow', 'future_order_hour_of_day']]

# Kami menggabungkan tabel 'users' dan 'us' dan menyimpan hasilnya ke dalam tabel 'users_features'
users_features = pd.merge(users , us, on='user_id', how='inner')

# Kami menghapus tabel 'us'
del us, users

users_features.head()

# 2.2 FITUR YANG BERGANTUNG PADA PRODUK

# Menghitung fitur-fitur yang bergantung pada produk
prod_features = orders_products.groupby(['product_id']).agg(
    prod_freq = ('order_id', 'count'),
    prod_avg_position = ('add_to_cart_order', 'mean')
#     prod_avg_hour = ('order_hour_of_day', average_times_of_day),
#     prod_avg_dow = ('order_dow', average_days)
).reset_index()

prod_features.head(20)

# Membuat kondisi yang menunjukkan apakah ini bukan pesanan pertama
non_first_order = orders_products['order_number'] != 1

# Menghitung rasio pemesanan ulang untuk setiap produk (produk_reorder_ratio)
grouped_orders_products = orders_products[non_first_order].groupby(['product_id']).agg(
    prod_reorder_ratio=('reordered', 'mean')
).reset_index()

# Menggabungkan hasil perhitungan ke dalam tabel 'prod_features'
prod_features = prod_features.merge(grouped_orders_products, on='product_id', how='left')

# Menghitung berapa kali pengguna membeli produk tersebut setelah pembelian pertama
grouped_orders_products = orders_products[non_first_order].groupby(['product_id', 'user_id']).agg(
    user_prod_freq=('order_id', 'count')
).reset_index()

# Menghitung rata-rata berapa kali pengguna membeli produk setelah pembelian pertama (user_prod_avg_freq)
grouped_orders_products = grouped_orders_products.groupby(['product_id']).agg(
    user_prod_avg_freq=('user_prod_freq', 'mean')
).reset_index()

# Menggabungkan hasil perhitungan ke dalam tabel 'prod_features'
prod_features = prod_features.merge(grouped_orders_products, on='product_id', how='left')

# Menghapus tabel-tabel yang sudah tidak diperlukan
del grouped_orders_products, non_first_order
prod_features.head()

# 2.3 PREDIKTOR PENGGUNA X PRODUK

# Membuat tabel data dengan mengelompokkan berdasarkan 'user_id' dan 'product_id'
data = orders_products.groupby(['user_id', 'product_id']).agg(
    up_orders=('product_id', 'count'),  # Jumlah total kali seorang pengguna memesan suatu produk
    up_first_order=('order_number', min),  # Waktu pertama seorang pengguna membeli suatu produk
    up_last_order=('order_number', max),  # Waktu terakhir seorang pengguna membeli suatu produk
    up_average_cart_position=('add_to_cart_order', 'mean')  # Rata-rata posisi dalam keranjang seorang pengguna dari suatu produk
).reset_index()

# Menghapus tabel 'orders_products' karena sudah tidak diperlukan
del orders_products

# Menampilkan 20 baris pertama dari tabel 'data'
data.head(20)

# Menghitung up_order_rate, yaitu rasio up_orders terhadap total produk yang dipesan oleh pengguna
data = data.merge(users_features[['user_id', 'user_orders']], on='user_id', how='left')
data['up_order_rate'] = data['up_orders'] / data['user_orders']

# Menghitung up_orders_since_last_order, yaitu selisih antara jumlah pesanan terakhir pengguna dan jumlah pesanan terakhir yang mencakup produk tersebut
data['up_orders_since_last_order'] = data['user_orders'] - data['up_last_order']

# Menghitung up_order_rate_since_first_order, yaitu rasio up_orders terhadap jumlah pesanan setelah pembelian pertama produk
data['up_order_rate_since_first_order'] = data['up_orders'] / (data['user_orders'] - data['up_first_order'] + 1)
# Ditambahkan +1 karena penomoran pesanan dimulai dari 1, bukan 0
del data['user_orders']

# Menampilkan hasil fitur tambahan terkait pengguna dan produk
data.head()

# Menggabungkan fitur-fitur pengguna dan produk dengan dataframe fitur akhir
data = data.merge(users_features, on='user_id', how='left').merge(prod_features, on='product_id', how='left')

# Menghapus tabel-tabel yang sudah tidak diperlukan
del users_features, prod_features

# Menggabungkan data pesanan masa depan (order_products_train) dengan data sebelumnya
order_products_future = order_products_train.merge(orders, on='order_id', how='left')
order_products_future = order_products_future[['user_id', 'product_id', 'reordered']]

# Menggabungkan data pesanan masa depan dengan data fitur
data = data.merge(order_products_future, on=['user_id', 'product_id'], how='left')

# Mengatur nilai 0 pada produk yang tidak ada dalam pesanan masa depan sehingga model dapat memprediksi bahwa produk tersebut tidak akan ada dalam pesanan masa depan.
data['reordered'].fillna(0, inplace=True)

# 2.4 PREDIKTOR YANG BERGANTUNG PADA WAKTU

'''
Menghitung perbedaan antara 2 nilai dari urutan berulang
dist(X, Y) = min { X-Y, N-(X-Y) }
'''
def diff_bet_time(arr1, arr2, max_value=23):
    arr1 = pd.to_datetime(arr1, format='%H')
    arr2 = pd.to_datetime(arr2, format='%H:%M:%S')
    arr_diff = np.abs(arr1.dt.hour-arr2.dt.hour)
    return np.minimum(arr_diff, max_value- (arr_diff-1))

'''
Menghitung perbedaan antara 2 nilai dari urutan berulang
dist(X, Y) = min { X-Y, N-(X-Y) }
'''
def diff_bet_dow(arr1, arr2, max_value=6):
    arr_diff = np.abs(arr1-arr2)
    return np.minimum(arr_diff, max_value- (arr_diff-1))

# data['up_hour_diff'] = diff_bet_time(data['future_order_hour_of_day'], data['up_avg_hour'])
# data['up_dow_diff'] = diff_bet_dow(data['future_order_dow'], data['up_avg_dow'])

# data['prod_hour_diff'] = diff_bet_time(data['future_order_hour_of_day'], data['prod_avg_hour'], )
# data['prod_dow_diff'] = diff_bet_dow(data['prod_avg_dow'], data['future_order_dow'])

# del data['prod_avg_dow'], data['prod_avg_hour'], data['future_order_hour_of_day'], data['up_avg_hour'], data['future_order_dow'], data['up_avg_dow']
del data['future_order_hour_of_day'], data['future_order_dow']

# 2.5 MEMBUAT X, Y

# Untuk menghemat memori, hapus kerangka data apa pun yang tidak akan kita gunakan selanjutnya
del order_products_prior, order_products_train, products, orders, departments, aisles

# Memisahkan data menjadi data pelatihan dan data uji
X_train = data[data['eval_set'] == 'train']
y_train = X_train['reordered']
X_test = data[data['eval_set'] == 'test']
del data


# Mengimpor modul train_test_split dari scikit-learn
from sklearn.model_selection import train_test_split

# Menampilkan distribusi kelas sebelum pemisahan
pos_count = np.sum(X_train['reordered'] == 1)
neg_count = np.sum(X_train['reordered'] == 0)
print('Rasio positif: ', pos_count)
print('Jumlah negatif: ', neg_count)
print('Rasio positif: ', pos_count / (pos_count + neg_count))

# Memisahkan data pelatihan menjadi data pelatihan dan data validasi dengan perbandingan 70% pelatihan dan 30% validasi
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)

# Menampilkan distribusi kelas dari set pelatihan
train_pos_count = np.sum(X_train['reordered'] == 1)
train_neg_count = np.sum(X_train['reordered'] == 0)
print('Jumlah positif: ', train_pos_count)
print('Jumlah negatif: ', train_neg_count)
print('Rasio positif: ', train_pos_count / (train_pos_count + train_neg_count))

# Menampilkan distribusi kelas dari set validasi
val_pos_count = np.sum(X_val['reordered'] == 1)
val_neg_count = np.sum(X_val['reordered'] == 0)
print('Jumlah positif: ', val_pos_count)
print('Jumlah negatif: ', val_neg_count)
print('Rasio positif: ', val_pos_count / (val_pos_count + val_neg_count))

# Menghapus kolom 'eval_set' dan variabel target dari fitur
X_train_non_pred_vars = X_train[['product_id', 'order_id', 'user_id']]
X_train.drop(['reordered', 'eval_set', 'product_id', 'order_id', 'user_id'], axis=1, inplace=True)

X_val_non_pred_vars = X_val[['product_id', 'order_id', 'user_id']]
X_val.drop(['reordered', 'eval_set', 'product_id', 'order_id', 'user_id'], axis=1, inplace=True)

X_test_non_pred_vars = X_test[['product_id', 'order_id', 'user_id']]
X_test.drop(['reordered', 'eval_set', 'product_id', 'order_id', 'user_id'], axis=1, inplace=True)

# Menghapus fitur yang dianggap redundan atau tidak signifikan berdasarkan analisis fitur importance
X_train.drop(['up_orders', 'up_last_order', 'user_total_products', 'user_distinct_products'], axis=1, inplace=True)
X_test.drop(['up_orders', 'up_last_order', 'user_total_products', 'user_distinct_products'], axis=1, inplace=True)
X_val.drop(['up_orders', 'up_last_order', 'user_total_products', 'user_distinct_products'], axis=1, inplace=True)

# Menghapus fitur yang bergantung pada waktu seperti 'up_dow_diff','prod_dow_diff','up_hour_diff','prod_hour_diff'
# Meskipun fitur-fitur waktu ini penting, hasil analisis fitur importance menunjukkan bahwa fitur ini memiliki kontribusi yang lebih rendah.
# Oleh karena itu, fitur ini dihapus.

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape)

X_train.columns

# 2.6 MELATIH MODEL

import xgboost as xgb
from sklearn import metrics

# Melatih model dengan fitur kecuali kolom product_id, user_id, dan order_id
clf = xgb.XGBClassifier(objective='binary:logistic', colsample_bytree = 0.4, learning_rate = 0.1,
                max_depth = 5, reg_lambda = 5.0, n_estimators = 100)
clf.fit(X_train,y_train)

import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel

# Memvisualisasikan tingkat kepentingan fitur
print(clf.feature_importances_)

xgb.plot_importance(clf)
plt.show()

# 2.7 ANALISIS PR CURVE

# Simpan probabilitas hanya untuk hasil positif
y_test_prob = clf.predict_proba(X_test)[:, 1]
y_val_prob = clf.predict_proba(X_val)[:, 1]
y_train_prob = clf.predict_proba(X_train)[:, 1]

'''
Fungsi ini memaksimalkan metrik tertentu, sambil menjaga metrik lain di atas ambang batas tertentu.
'''
def maximize_metric_keep_metric(metric1_list, metric2_list, metric2_thresh=0.3):
    for idx in range(len(metric1_list)):
        if metric2_list[idx] > metric2_thresh:
            return idx
    return -1

from sklearn.metrics import precision_recall_curve

# Memilih ambang batas yang memaksimalkan f1_score
precision, recall, thresholds = precision_recall_curve(y_val, y_val_prob)
f1_scores = 2*recall*precision/(recall+precision)
opt_indx = np.argmax(f1_scores)
print("Nilai f1_score maksimum untuk kelas positif: ", f1_scores[opt_indx])
print("Presisi yang sesuai: ", precision[opt_indx])
print("Recall yang sesuai: ", recall[opt_indx])
print("Ambang batas yang sesuai: ", thresholds[opt_indx])
best_thresh = thresholds[opt_indx]

# Memilih ambang batas yang memaksimalkan recall, sambil menjaga presisi di atas 0.3
opt_indx = maximize_metric_keep_metric(metric1_list=recall, metric2_list=precision, metric2_thresh=0.3)
print("Recall maksimum untuk kelas positif: ", recall[opt_indx])
print("Presisi yang sesuai: ", precision[opt_indx])
print("Nilai f1_score yang sesuai: ", f1_scores[opt_indx])
print("Ambang batas yang sesuai: ", thresholds[opt_indx])
best_thresh = thresholds[opt_indx]

# Memplot kurva presisi-recall
no_skill = len(y_val[y_val==1]) / len(y_val)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(recall, precision, marker='.', label='Logistik')
plt.plot(recall[opt_indx], precision[opt_indx], marker='o', color='k', label='ambang batas optimum')
# label sumbu
plt.xlabel('Recall')
plt.ylabel('Presisi')
plt.title('Kurva Presisi-Recall')
# tampilkan legenda
plt.legend()
# tampilkan plot
plt.show()

# Mengubah probabilitas menjadi nilai prediksi tegas, dengan menggunakan ambang batas yang diperoleh dari kurva ROC
y_test_preds = y_test_prob > best_thresh
y_val_preds = y_val_prob > best_thresh
y_train_preds = y_train_prob > best_thresh

# 2.8 CLASSIFICATION REPORT

from sklearn.metrics import confusion_matrix, classification_report

print('-----------------LAPORAN KLASIFIKASI--------------------')
print("Jumlah kelas positif pada Data Latih: ", y_train.sum())
print("Jumlah kelas negatif pada Data Latih: ", y_train.shape[0] - y_train.sum())
print("Data Latih tn, fp, fn, tp:", confusion_matrix(y_train, y_train_preds).ravel())
print("Laporan Data Latih:", classification_report(y_train, y_train_preds))

print("Jumlah kelas positif pada Data Validasi: ", y_val.sum())
print("Jumlah kelas negatif pada Data Validasi: ", y_val.shape[0] - y_val.sum())
print("Data Validasi tn, fp, fn, tp:", confusion_matrix(y_val, y_val_preds).ravel())
print("Laporan Data Validasi:", classification_report(y_val, y_val_preds))

# 2.9 PREPARE SUBMISSION FILE

import csv

# Menambahkan prediksi ke detail pesanan uji
test_orders = X_test_non_pred_vars[['order_id','product_id']]
test_orders['reordered'] = y_test_preds

# Mengekstrak pesanan yang tidak memiliki produk yang diprediksi
empty_orders = test_orders.groupby(['order_id']).agg(
    count_reorders = ('reordered', 'sum')
).reset_index()
empty_orders = empty_orders[empty_orders['count_reorders'] == 0]

# Untuk pesanan yang memiliki produk yang diprediksi
# Ekstrak produk yang diprediksi akan ada di pesanan mendatang
test_orders = test_orders[test_orders['reordered'] == 1]
# Untuk setiap pesanan, kelompokkan produk yang diprediksi bersama ke dalam daftar
test_orders = test_orders.groupby('order_id')['product_id'].apply(list).reset_index(name='products')


test_orders.head()

# csv header
headerNames = ['order_id', 'products']
rows = []

for index, row in test_orders.iterrows():
    products = ' '.join(str(product_id) for product_id in row['products'])
    rows.append(
        {'order_id': str(row['order_id']),
        'products': products})

for index, row in empty_orders.iterrows():
    rows.append(
        {'order_id': str(row['order_id']),
        'products': 'None'})

with open('./submissions.csv', 'w', encoding='UTF-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headerNames)
    writer.writeheader()
    writer.writerows(rows)