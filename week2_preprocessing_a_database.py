# -*- coding: utf-8 -*-
"""Week2-preprocessing a database.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1g7Lvecjly8ex2wN1BHxNdacRbO7PvDDf

Exercise Week 2 - Processing Data

The sample database simulates case records of people who have been
infected with COVID-19 in Brazil. The base data dictionary is as follows:
• id: record identifier field;
• idade = age: age of the person;
• uf: state where the person was diagnosed;
• renda = income: social class to which the person belongs, varying between A and E;
• vacina = vaccine: indicates whether the person has been vaccinated (1) or not (0).
"""

import pandas as pd

url = 'https://raw.githubusercontent.com/higoramario/univesp-com360-mineracao-dados/main/dados-covid-limpeza.csv'

cases_covid = pd.read_csv(url)

cases_covid.head(20)

cases_covid.info()

cases_covid.describe()

cases_covid['uf'].value_counts()   #SP is the same state that sp

cases_covid['uf'] = cases_covid['uf'].str.upper() # All capital letters
cases_covid['uf'].value_counts()

cases_covid['renda'].value_counts()

cases_covid['vacina'].value_counts()

cases_covid['idade'].fillna(round(cases_covid['idade'].mean()), inplace = True) #fills the null values ​​in the age column with the mean
cases_covid['idade'].value_counts()

cases_covid['uf'].fillna(method='ffill', inplace = True) # fill method fills null values ​​with the next valid values
cases_covid['uf'].value_counts()

cases_covid['renda'].fillna(method = 'bfill', inplace = True) # bfill method fills the null values ​​with the previous object
cases_covid['renda'].value_counts()

cases_covid.info()

cases_covid.head(len(cases_covid))

