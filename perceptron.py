# -*- coding: utf-8 -*-

#Neurona
import csv
import numpy as np
from numpy import genfromtxt

w=[]
for i in range(0,3):
  n=np.random.randint(-10,10)
  w.append(n)
alpha = 0.1
print("Pesos iniciales: ", w)

prototipos = genfromtxt('Data1_entrenamiento.csv', skip_header=1, delimiter=",", usecols=(0,1,2))

x = genfromtxt('Data1_validacion.csv', skip_header=1, delimiter=",", usecols=(0,1,2))

etiqueta_prototipos = genfromtxt('Data1_entrenamiento.csv', skip_header=1, delimiter=",", usecols=(3))

etiquetas_validacion = genfromtxt('Data1_validacion.csv', skip_header=1, delimiter=",", usecols=(3))

#t son las clases
def perceptron(prototipos, etiquetas_prototipos, w, alpha):
  epocas = 0
  while True: 
    E = 0
    for index, p in enumerate(prototipos):
      a=np.dot(p,w)
      if(a>=0):
        y=1
      else:
        y=0
      if etiquetas_prototipos[index]!=y: 
        w=w+alpha*(etiqueta_prototipos[index]-y)*prototipos[index]
      E=E+etiquetas_prototipos[index]-y
      epocas+=1
    if E==0:
      break
  return w

#t son las clases
def perceptron_validacion(x,w):
  clasificacion = []
  for p in x:
    a=np.dot(p,w)
    if(a>=0):
      y=1
    else:
      y=0
    clasificacion.append(y)
  return clasificacion

aciertos = sum(x == y for x, y in zip(perceptron_validacion(x,w), etiquetas_validacion))
print(aciertos)

w = perceptron(prototipos, etiqueta_prototipos, w, alpha)
print("Pesos resultantes: ", w)

print("Rendimiento: ", aciertos/len(etiquetas_validacion)*100,"%")