# -*- coding: utf-8 -*-
"""week6_association_rule_learning_Apriori.ipynb

Data Mining - Week 6:
         
         Binding Rules Exercise
        
         In this exercise, we will use the **Apriori** algorithm on a database containing supermarket purchase transactions.
      
         Data:
        
         * Each row represents a transaction and each column corresponds to a purchased product. The cells contain a sequence of products purchased in each transaction, otherwise the value is null.
       
         Data was obtained from Kaggle https://www.kaggle.com/code/timothyabwao/market-basket-analysis-using-the-apriori-algorithm/data and is available at this link https: //raw.githubusercontent.com/higoramario/univesp-com360-mineracao-dados/main/market-basket-optimisation.csv)

### Libraries and database
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["figure.figsize"] = (15,10)

!pip -qq install apyori
import apyori

# Each object represents a purchase or transaction, and contains the name of one or more products that are part of the purchase.

url = 'https://raw.githubusercontent.com/higoramario/univesp-com360-mineracao-dados/main/market-basket-optimisation.csv'
market = pd.read_csv(url, header=None)
market.head(10)

len(market)

for index in market.columns:
  market[index] = market[index].str.strip() # Remove values with right or left spaces using the strip function.

items = market.melt()['value'].dropna().sort_values() # How many different products are in the base and what they are. Melt function: join values. Sort_values function: sort
print('There are {} different products:\n {}'.format(items.nunique(), items.unique()))

quantity_items = items.value_counts()

bar = quantity_items.nlargest(10).plot(kind = 'bar') 
bar.set_title('Best Sellers', size = 20, weight = 500, pad = 15)
bar.set_ylabel('Quantity')
plt.show()

quantity_items = items.value_counts()

bar = quantity_items.nsmallest(10).plot(kind = 'bar') 
bar.set_title('Least Sellers', size = 20, weight = 500, pad = 15)
bar.set_ylabel('Quantity')
plt.show()

quantity_per_basket = market.notna().apply(sum, axis = 1) # Number of non-null items

baskets = [set(row.dropna()) for _, row in market[quantity_per_basket > 1].iterrows()] # Items that have more than one product per transaction.
baskets[:5]

len(baskets)

"""### Apriori"""

# # A => B [support, confidence], where set A is called the antecedent rule and set B is called consequent rule.
# Support: how many transactions should I consider reasonable to say that a pattern is important. 
# Confidence: it's the proportion of times a transaction contains element A and also element B
# The smaller the support value, the greater the number of combinations and consequently the computational cost.

# Minimum 4% support because the base is dispersed. 30% confidence
minsup = 0.04 
minconf = 0.3

processing_rules = apyori.apriori(baskets, min_support = minsup, min_confidence = minconf)
print('Item and consequent item:')
for rule in processing_rules:
  items = list(rule.items)
  print('{} --> [{}] Support: {:.3f}. Confidence: {:.3f}'.format(items[:-1], items[-1], rule.support, rule.ordered_statistics[0].confidence))