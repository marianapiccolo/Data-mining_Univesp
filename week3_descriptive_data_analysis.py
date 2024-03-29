# -*- coding: utf-8 -*-
"""week3_descriptive_data_analysis.ipynb

Automatically generated by Colaboratory.


The following dataset contains information about babies born and their parents. 

URL do conjunto de dados:
        https://www.sheffield.ac.uk/polopoly_fs/1.937185!/file/Birthweight_reduced_kg_R.csv,
      Mathematics and Statistics Help (MASH), Universidade de Sheffield, Reino Unido."

         "1. Make a histogram of the baby's weight and baby's height attributes.",
         "2. Generate a pie chart of smoking mothers.",
         "3. Get the mean and standard deviation of the babies' heights.,
         "4. Generate a box plot of the mother's age.",
         "5. Generate a scatterplot of baby weight and gestation time.",
         "6. Add information about smoking and non-smoking mothers. To do so, include the **hue** parameter in the **scatterplot** function, passing the **smoker** attribute as a reference."

Description of attributes:

        "**ID**: baby's number,
        "**Length**: baby's length (cm),
        "**Birthweight**: baby weight (kg),
        "**Headcirc**: head circumference,
        "**Gestation**: gestation (weeks),
        "**smoker**: mother smoker (1 = yes, 0 = no),
        "**mage**: mother's age,
        "**mnocig**: number of cigarettes per mother's day,
        "**mheight**: mother's height (cm),
        "**mppwt**: mother's pre-pregnancy weight (kg),
        "**fage**: father's age,
        "**fedyrs**: father's years of schooling,
        "**fnocig**: number of cigarettes per father's day,
        "**fheight**: height of parent (cm),
        "**lowbwt**: low birth weight (0 = no, 1 = yes),
        "**mage35**: mother over 35 (0 = no, 1 = yes).
"""

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

sb.set(rc={'figure.figsize':(15,8)})

url = 'https://www.sheffield.ac.uk/polopoly_fs/1.937185!/file/Birthweight_reduced_kg_R.csv'
babies = pd.read_csv(url)
babies.head()

# Histogram of the baby's weight and baby's height
sb.displot(babies['Birthweight'])
plt.show() #weight

sb.displot(babies['Length']) #height
plt.show()

# Pie chart of smoking mothers
smokers = len(babies[babies['smoker']==1])
not_smokers = len(babies[babies['smoker']==0])

plt.pie([smokers, not_smokers], labels=['Smokers', 'Not smokers'], autopct = '%0.0f%%')

# Mean and standard deviation of the babies' heights
babies['Length'].describe()

# Box plot of the mother's age
sb.boxplot(y=babies['mage'])
plt.show()

# Scatterplot of baby weight and gestation time
# 1 = smoker; 2 = not smoker
sb.scatterplot(x=babies['Birthweight'], y=babies['Gestation'], hue=babies['smoker'])
plt.show()

# We can see in the scatter plot that all babies weighit less than 2.5 kg are born to smoking mothers. 
# There is also a higher concentration of babies with less weight among smoking mothers.
# Another visible (and expected) result is that there is a positive correlation between the time of gestation and the weight of the babies."