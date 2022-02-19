from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from db import getC1ItemSets, getNumUniqueItems
import pandas as pd

'''
************************
Approach to use a library to generate association rules:
https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/
************************
 
'''

dataset = getC1ItemSets('track_uri', 100)  # Doesnt even work for 100 playlists and minSup 0.02
lenUniqueTracks = getNumUniqueItems('track_uri', 100)

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequentItemSets = fpgrowth(df, min_support=0.02, use_colnames=True)
rules = association_rules(frequentItemSets, metric="confidence", min_threshold=0.7)
antecedents = set()
for antecedent in rules['antecedents']:
    antecedents.add(antecedent)
print(rules['antecedents'])
print(f"{len(antecedents)} antecedents from {lenUniqueTracks} ({round(len(antecedents) / lenUniqueTracks * 100, 2)}%)")

# print(apriori(df, 0.01, low_memory=False))
