#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:51:48 2017

@author: Claudio
"""
import psycopg2
import matplotlib.pyplot as plt
from matplotlib.pylab import *
import scipy
from pandas import DataFrame
from bisect import bisect_left
import numpy as np
from scipy import stats
from scipy.integrate import simps
import pylab as pl


conn = psycopg2.connect("host='localhost' dbname='datos' user='postgres' password='123456789'")
cursor = conn.cursor()

condicionT = cursor.execute("SELECT cuenta, archivo, rt, condicion FROM datos WHERE congruency = 'incongruent'  AND response = 'OK' AND condicion = 'T' and archivo = 'ivan_ST'")#azul
condicionT = cursor.fetchall()


condicionC = cursor.execute("SELECT cuenta, archivo, rt, condicion FROM datos WHERE congruency = 'incongruent'  AND response = 'OK' AND condicion = 'C'  and archivo = 'ivan_SC'")#azul
condicionC = cursor.fetchall()

condicionH = cursor.execute("SELECT cuenta, archivo, rt, condicion FROM datos WHERE congruency = 'incongruent'  AND response = 'OK' AND condicion = 'H'  and archivo = 'ivan_SH'")#azul
condicionH = cursor.fetchall()

condicionV = cursor.execute("SELECT cuenta, archivo, rt, condicion FROM datos WHERE congruency = 'incongruent'  AND response = 'OK' AND condicion = 'V' AND  archivo = 'ivan_SV'")#azul
condicionV = cursor.fetchall()


print len(condicionH)

#condicionA = cursor.execute("SELECT ROW_NUMBER() OVER (ORDER BY archivo), archivo, CASE condicion WHEN 'C' THEN rt ELSE NULL END cond_C, CASE condicion WHEN 'T' THEN rt ELSE NULL END cond_T, CASE condicion WHEN 'V' THEN rt ELSE NULL END cond_V, CASE condicion WHEN 'H' THEN rt ELSE NULL END cond_H FROM datos WHERE condicion IN ('C', 'T', 'V', 'H') AND response = 'OK' ORDER BY archivo")
#condicionA = cursor.fetchall()

print condicionV







