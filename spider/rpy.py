#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:48:58 2017

@author: Claudio
"""
import numpy as np, scipy.stats, pandas as pd
import psycopg2
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import rpy2
import pylab as pl
import ipython

#%matplotlib inline

conn = psycopg2.connect("host='localhost' dbname='datos' user='postgres' password='123456789'")
cursor = conn.cursor()

condT = cursor.execute("SELECT cuenta, archivo, rt, condicion FROM datos WHERE congruency = 'incongruent'  AND response = 'OK' AND condicion = 'T' and archivo = 'ivan_ST'")#azul
condT = cursor.fetchall()

condC = cursor.execute("SELECT cuenta, archivo, rt, condicion FROM datos WHERE congruency = 'incongruent'  AND response = 'OK' AND condicion = 'C'  and archivo = 'ivan_SC'")#azul
condC = cursor.fetchall()

condH = cursor.execute("SELECT cuenta, archivo, rt, condicion FROM datos WHERE congruency = 'incongruent'  AND response = 'OK' AND condicion = 'H'  and archivo = 'ivan_SH'")#azul
condH = cursor.fetchall()

condV = cursor.execute("SELECT cuenta, archivo, rt, condicion FROM datos WHERE congruency = 'incongruent'  AND response = 'OK' AND condicion = 'V' AND  archivo = 'ivan_SV'")#azul
condV = cursor.fetchall()

random.seed(1) #seed random number generator
cond_1 = [random.gauss(600,30) for x in range(30)] #condition 1 has a mean of 600 and standard deviation of 30
cond_2 = [random.gauss(650,30) for x in range(30)] #u=650 and sd=30
cond_3 = [random.gauss(600,30) for x in range(30)] #u=600 and sd=30

plt.bar(np.arange(1,4),[np.mean(cond_1),np.mean(cond_2),np.mean(cond_3)],align='center') #plot data
plt.xticks([1,2,3]);

load_ext rpy2.ipython

#pop the data into R
Rpush cond_1 cond_2 cond_3

#label the conditions
R Factor <- c('Cond1','Cond2','Cond3')
#create a vector of conditions
R idata <- data.frame(Factor)

#combine data into single matrix
R Bind <- cbind(cond_1,cond_2,cond_3)
#generate linear model
R model <- lm(Bind~1)

#load the car library. note this library must be installed.
R library(car)
#run anova
%R analysis <- Anova(model,idata=idata,idesign=~Factor,type="III")
#create anova summary table
%R anova_sum = summary(analysis)

#move the data from R to python
%Rpull anova_sum
print anova_sum
print ','.join(map(str,condV[2]))

#print condV[2]
