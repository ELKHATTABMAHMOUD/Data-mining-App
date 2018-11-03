#!/usr/bin/env python
# -*- coding: utf-8 -*-


from scipy.io import arff
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# QUESTION 1
# les valeurs manquantes sont représentées par des ?. on propose de les remplacer par des 0. 
data = arff.loadarff('vote.arff')
votes = pd.DataFrame(data[0])
#print(votes)
vote_one_hot = pd.get_dummies(votes)
vote_one_hot.drop(vote_one_hot.filter(regex='_\?$',axis=1).columns,axis=1,inplace=True)

#print(vote_one_hot)

# QUESTION 2 
frequent_itemsets = apriori(vote_one_hot, min_support=0.3, use_colnames=True)

# Statistiques en fonction du nombre d'items
frequent_itemsets['Nombre items'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets)


# QUESTION 3
# le nombre de règles est le nombre de lignes, comenter la première règle obtenu 
association_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9)
#print(association_rules)


# QUESTION 4 
# Voir Rapport 


# QUESTION 5

