'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random
from math import log, e

# PES2UG19CS064 Arnav Dewan

'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):
    last_col = df.iloc[:,-1:]
    values = last_col.value_counts()
    values_reduced = values / sum(values)
 
    entropy = 0
    base = 2
    for i in values_reduced:
        entropy -= i * log(i, base)
    return entropy


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):
    values = df[attribute].unique()
    values_count = df[attribute].value_counts()
    sum_count = sum(values_count)
    avg_entropy_children = 0
    for value in values:
        avg_entropy_children += (values_count[value] / sum_count) * get_entropy_of_dataset(df.loc[df[attribute] == value])
    return avg_entropy_children


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):
    return get_entropy_of_dataset(df) - get_avg_info_of_attribute(df, attribute)


#input: pandas_dataframe
#output: ({dict},'str')
def get_selected_attribute(df):
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
    attributes_list = [attribute for attribute in df][:-1]
    ig_dict = { attribute:get_information_gain(df, attribute) for attribute in attributes_list}
    return (ig_dict, max(ig_dict, key=lambda x: ig_dict[x]))
