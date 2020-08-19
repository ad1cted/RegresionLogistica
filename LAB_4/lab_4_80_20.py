import pandas as pd
from math import sqrt
#import numpy as np
import matplotlib.pyplot as plt



#funcion para generar dumy
#variables dummy para genero
def create_dummy(df,elemento):
    genero_dummy = pd.get_dummies(df[elemento], prefix=elemento)
    df = df.merge(genero_dummy,left_index=True,right_index=True)
    df = df.drop(columns=elemento)
    return df


###### Preprocesamiento

df =  pd.read_csv("/home/eplaza/Insync/esteban.plaza.a@usach.cl/Google Drive/USACH/Analisis_de_Datos/Laboratorios/LAB 4/datapreprocesadaBruto.csv", sep=",")
a=df
clase = df["aprobado"]   #y


clase = clase.replace('aprobado', 1)
clase = clase.replace('noAprobado', 0)

#df = df[["tiene_mora","salario","score_crediticio","etnia","nivel_educacion","anios_empleado","otros_ingresos"]]
#df = df[["licencia_conducir"]]

df = df.drop(columns="aprobado") #x
df = create_dummy(df,"genero")
df = create_dummy(df,"nivel_educacion")
df = create_dummy(df,"estado_civil")
df = create_dummy(df,"etnia")
df = create_dummy(df,"empleado")
df = create_dummy(df,"ciudadania")
df = create_dummy(df,"tipo_cliente")
df = create_dummy(df,"tiene_mora")
df = create_dummy(df,"licencia_conducir")





####### Analisis


# dividimos el dataset  80 % entrenamiento 20% de testeo
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df,clase, test_size=0.1, random_state = 0)


# escalamos los datos
from sklearn.preprocessing import StandardScaler
escalador = StandardScaler()
x_train = escalador.fit_transform(X=x_train)
x_test = escalador.transform(X=x_test )



#regresion logistica
from sklearn.linear_model import LogisticRegression
logistic_regresion =  LogisticRegression(max_iter=10000, n_jobs=-1)


logistic_regresion.fit(x_train, y_train)



## odds ratio
odds_ratio = pd.DataFrame(logistic_regresion.coef_[0])
columns_odds = pd.DataFrame(df.columns)
odds = pd.concat([odds_ratio,columns_odds],join="inner", axis = 1, ignore_index=True)
odds.columns = ["ratio", "variable"]
odds =odds.sort_values("ratio",ascending=False)


# hacemos una prediccion con los datos de testeo
y_pred = logistic_regresion.predict(x_test)
porcenaje_predict  = pd.DataFrame(logistic_regresion.predict_proba(x_test)) 



# MATRIZ DE CONFUSION
from sklearn.metrics import  plot_confusion_matrix
plot_confusion_matrix(logistic_regresion,x_test, y_test)





##CURVA ROC
from sklearn.metrics import roc_curve

false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred)
plt.subplots(1, figsize=(10,10))
plt.title('Curva roc ')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('Verdaderos positivos Rate')
plt.xlabel('Falsos Positivos Rate')
plt.show()



# calculamos algunas metricas de calidad
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
print("Precision Score -> {}".format(precision_score(y_test, y_pred)))
print("Accuracy -> {}".format(accuracy_score(y_test, y_pred)))
print("Recall -> {}".format(recall_score(y_test, y_pred)))
print("Mean Squared Error(MSE) -> {}".format(mean_squared_error(y_test, y_pred)))
print("Root-Mean-Squared-Error(RMSE) -> {}".format(sqrt(mean_squared_error(y_test, y_pred))))
print("Mean-Absolute-Error(MAE) -> {}".format(mean_absolute_error(y_test, y_pred)))
print("R² -> {}".format(r2_score(y_test, y_pred)))






