import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def load_and_process_data():
    # Load your data
    data_directory_path = 'csv/'

    order_products_train = pd.read_csv(data_directory_path + 'order_products__train.csv')
    orders = pd.read_csv(data_directory_path + 'orders.csv')
    products = pd.read_csv(data_directory_path + 'products.csv')

    # Merge order_products_train with products based on product_id
    merged_df = pd.merge(order_products_train, products, on='product_id', how='left')

    # Merge the result with orders based on order_id
    df = pd.merge(merged_df, orders, on='order_id', how='left')

    df = df[df['eval_set'] == 'train']

    # Process the data and perform Apriori algorithm
    product_counts = df.groupby('product_id')['order_id'].count().reset_index().rename(columns={'order_id': 'frequency'})
    product_counts = product_counts.sort_values('frequency', ascending=False).head(100).reset_index(drop=True)

    freq_products = list(product_counts.product_id)
    del product_counts

    order_products = df[df.product_id.isin(freq_products)]
    del df

    df = order_products[['order_id', 'product_name', 'reordered']].set_index('order_id')

    basket = df.pivot_table(columns='product_name', values='reordered', index='order_id').reset_index().fillna(0).set_index('order_id')

    def encode_units(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1

    basket = basket.applymap(encode_units)
    shortbasket = basket[:100000]

    frequent_items = apriori(shortbasket, min_support=0.001, use_colnames=True, verbose=1, low_memory=True)
    frequent_items['length'] = frequent_items['itemsets'].apply(lambda x: len(x))

    rules = association_rules(frequent_items, metric='lift', min_threshold=1.1)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))

    return frequent_items, rules

def select_rules_with_antecedents_names(rules, names=set()):
    return rules[rules['consequents'].apply(lambda x:  names in {x})]