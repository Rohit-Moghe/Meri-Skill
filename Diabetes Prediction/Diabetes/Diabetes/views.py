from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from django.shortcuts import render

def result(request):
  df = pd.read_csv("/content/diabetes.csv")
  X = df.drop('Outcome', axis = 1)
  Y = df['Outcome']
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

  LOR = LogisticRegression()
  LOR.fit(X_train, Y_train)

  val1 = float(request.GET['n1'])
  val2 = float(request.GET['n2'])
  val3 = float(request.GET['n3'])
  val4 = float(request.GET['n4'])
  val5 = float(request.GET['n5'])
  val6 = float(request.GET['n6'])
  val7 = float(request.GET['n7'])
  val8 = float(request.GET['n8'])

  pared = LOR.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

  result = ""
  if pared == [1]:
    result1 = "Positive"
  elif pared == [0]:
    result1 = 'Negative'
  
  return render(request, "predict.html", {"result2": result1})