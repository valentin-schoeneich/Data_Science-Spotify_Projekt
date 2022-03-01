from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules, apriori
from db import getC1ItemSets, getNumUniqueItems
import pandas as pd

'''
************************
Approach to use a library to generate association rules:
https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/
************************
'''


def apriori1():
    dataset = getC1ItemSets('track_uri', 100)  # Doesnt even work for 100 playlists and minSup 0.02
    lenUniqueTracks = getNumUniqueItems('track_uri', 100)

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequentItemSets = fpgrowth(df, min_support=0.02, max_len=2, use_colnames=True)
    print(frequentItemSets)
    rules = association_rules(frequentItemSets, metric="confidence", min_threshold=0.7)
    antecedents = set()
    for antecedent in rules['antecedents']:
        antecedents.add(antecedent)
    print(rules['antecedents'])
    print(f"{len(antecedents)} antecedents from {lenUniqueTracks} ({round(len(antecedents) / lenUniqueTracks * 100, 2)}%)")


def apriori2():
    # testing mlxtend with small and clear data

    dataset = [['Track0', 'Track1', 'Track2', 'Track3', 'Track4', 'Track5'],
               ['Track2', 'Track6', 'Track7', 'Track4', 'Track8', 'Track5'],
               ['Track6', 'Track1', 'Track2', 'Track7'],
               ['Track6', 'Track4', 'Track0', 'Track8']]

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequentItemSets= fpgrowth(df, min_support=0.5)
    print(frequentItemSets)
    rules = association_rules(frequentItemSets, metric="confidence", min_threshold=0.7)
    print(rules)

apriori2()
