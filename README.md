# Datascience
import pandas as pd
import numpy as np
import matplotlib as pyplot
import seaborn as sns
cereal=pd.read_csv("F:\material\data\edureka\cereal.csv")
cereal.info()
cereal.describe()
correlations=cereal.corr()
correlations
sns.heatmap(correlations)
sns.pairplot(cereal,hue="rating")
cereal.head(5)
y=cereal["rating"]
x=cereal.drop("rating",axis=1)
from sklearn.model_selection import train_test_split
trainx, testx, testy, trainy = train_test_split(x,y, test_size=0.3,random_state=101)
trainx.head()
trainx.shape
trainy.shape
testx.shape
testy.shape
from sklearn.linear_model import LinearRegression
linearreg = LinearRegression()
LinearRegression.fit(trainx,trainy)
predictedop=linearreg.predict(testx)
predictedop
testy
pd.cereal({"true label":testy,"predictedop":predictedop})
from sklearn.metrics import r2_score
linearreg.coef_
cereal.head()
linearreg.intercept_
