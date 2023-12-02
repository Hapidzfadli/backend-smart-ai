# Install library mlxtend (jika belum terinstall)
# pip install mlxtend

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Dataset
data = [
    ["Banna", "Apple Honeycrisp Organic", "Asparagus", "Banana", "Boneless Skinless Chicken Breasts"],
    ["Broccoli Crown", "Banna", "Asparagus", "Carrots", "Bunched Cilantro", "Cucumber Kirby", "Boneless"],
    ["Asparagus", "Carrots", "Banna", "Apple", "Skinless", "Broccoli Crown", "Limes"]
]

# Menggunakan TransactionEncoder untuk mengonversi dataset menjadi format yang sesuai dengan algoritma Apriori
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Menerapkan algoritma Apriori untuk mendapatkan itemset yang sering muncul
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

# Menerapkan association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Menampilkan rules yang memuat "Banna" dan "Apple" di antara antecedents
filtered_rules = rules[rules['antecedents'].apply(lambda x: "Skinless" in x and "Apple" in x)]

# Menampilkan barang yang mungkin dibeli berikutnya
predicted_items = set(filtered_rules['consequents'].explode())
print("Barang yang mungkin dibeli berikutnya:", predicted_items)
