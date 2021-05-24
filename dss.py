import pandas as pd
import numpy as np

def remove_list_item (*, the_list, the_item):
  new_list = [item for item in the_list if item != the_item]
  return new_list


def plot_x_by_class_y(*, table, x_column, y_column):
  assert isinstance(table, pd.core.frame.DataFrame), f'table is not a dataframe but instead a {type(table)}'
  assert x_column in table.columns, f'unrecognized column: {x_column}. Check spelling and case.'
  assert y_column in table.columns, f'unrecognized column: {y_column}. Check spelling and case.'
  assert table[y_column].nunique()<=5, f'y_column must be of 5 categories or less but has {table[y_column].nunique()}'

  pd.crosstab(table[x_column], table[y_column]).plot(kind='bar', figsize=(15,8), grid=True, logy=True)
  return None

def percent_change (*, new, old):
  change = (new - old)/old
  return change

def replace_high_outliers (*, df, column):
  sigma = np.std(df['column'])
  sigma3 = sigma * 3
  high_wall = df['column'].mean() + sigma3
  column_list = df['column'].tolist()
  cleaned_column = [min(item, high_wall) for item in column_list]
  df['column']= cleaned_column
  return None

def decision_rule(*, triple):
  actual = triple[2]
  c_0_score = triple[0][1]
  prediction = 0 if c_0_score > 0 else 1  #ternary conditional
  return [prediction, actual]

def mcc(*, tp, tn, fp, fn):
  denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
  mcc_score = 0 if denom==0 else (tp*tn-fp*fn)/denom**.5 #gets around divide by zero error
  return mcc_score

def wrangle_text(*, essay):
  assert isinstance(essay, str) == True
  doc = nlp(essay)
  string_essay = [item.text.lower() for item in doc if item.is_alpha and not item.is_oov and not item.is_stop]
  return string_essay

def accuracy(*, tp, tn, fp, fn):
  denom = (tp+tn+fp+fn)
  acc = 0 if denom==0 else (tp+tn)/denom
  return acc

def precision(*, tp, fp):
  denom = tp+fp
  prec = 0 if denom==0 else (tp/denom)
  return prec

def recall(*, tp, fn):
  denom = tp+fn
  rec = 0 if denom==0 else (tp/denom)
  return rec

def f1_score(*, p, r):
  return 2*p*r/(p+r)

def mcc_from_pairs(*, p_a_pairs):
  mscore = mcc(tp=p_a_pairs.count([1,1]),
                tn=p_a_pairs.count([0,0]),
                fp=p_a_pairs.count([1,0]),  #predicted 1 but actual 0, i.e., FP
                fn=p_a_pairs.count([0,1]))
  return mscore

def f1_from_pairs(*, p_a_pairs):
  f1score = f1_score(p = precision(tp=p_a_pairs.count([1,1]), fp = p_a_pairs.count([1,0])), r =recall(tp=p_a_pairs.count([1,1]), fn = p_a_pairs.count([0,1])))
  return f1score
