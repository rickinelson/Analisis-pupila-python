#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 20:57:38 2017

@author: Claudio
"""


import psycopg2
import matplotlib.pyplot as plt
from matplotlib.pylab import *
import scipy
from scipy.signal import butter, filtfilt
from pandas import DataFrame
from bisect import bisect_left
import numpy as np
from scipy import stats
from scipy.integrate import simps

def butter_lowpass_filter(data, highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='lowpass')
    y = filtfilt(b, a, data)
    return y


def blinkfix(trial, bound=50, highcut=2., fs=500., order=5, threshold=1):
    trial[np.isnan(trial)] = 0
    
    mask = np.diff((trial < threshold).astype(np.int))
    heads = np.flatnonzero(mask == 1)     # start points.
    tails = np.flatnonzero(mask == -1)    # end points.

    if (heads.size != 0) and (tails.size != 0):
        if tails[0] < heads[0]:
            heads = np.hstack((0, heads))
        if tails[-1] < heads[-1]:
            tails = np.hstack((tails, trial.size - 1))

        heads -= bound
        tails += bound

        mask = np.zeros(trial.shape, dtype=np.bool)
        for a, b in zip(heads, tails):
            mask[a:b] = True
        mask = ~mask

        trial = scipy.interpolate.splev(
            np.arange(trial.size), scipy.interpolate.splrep(
                    np.flatnonzero(mask), trial[mask], k=3
                )
        )
        
        trial = np.interp(
            np.arange(trial.size), np.flatnonzero(mask), trial[mask]
        )

    return butter_lowpass_filter(trial, highcut, fs, order=3)# comando para filtar por frecuencia
#    return (trial)# comando para mostrar la senal cruda



conn = psycopg2.connect("host='localhost' dbname='datos' user='postgres' password='123456789'")
cursor = conn.cursor()


ARCHIVO = 'nico_PH'

x = "SELECT tiempo, mensaje FROM mensajes WHERE archivo = '%s' AND mensaje LIKE '%%trial: 1%%' ORDER BY tiempo" % (ARCHIVO)
cursor.execute(x)
fin_estimulacion = cursor.fetchall()[0]

y = "SELECT onset, pup_l, pup_r FROM pupila WHERE onset BETWEEN %d - 2500 AND %d AND archivo = '%s' ORDER BY onset" % (fin_estimulacion[0], fin_estimulacion[0], ARCHIVO)
cursor.execute(y)
pupila = cursor.fetchall()


    
if len(pupila) > 20000: exit() 
if len(pupila) == 0: exit()

cantidad_l_cero = len([pup[1] for pup in pupila if pup[1] == 0.0])
cantidad_r_cero = len([pup[2] for pup in pupila if pup[2] == 0.0])
        	
porcentaje_l_cero = 100*float(cantidad_l_cero)/len(pupila)
porcentaje_r_cero = 100*float(cantidad_r_cero)/len(pupila)

onsets = [pup[0] for pup in pupila]

pupilas = [float(pup[1])     for pup in pupila] #revisar si es la mejor estrategia, ademas si promedias debiera ser despues del filtro
        
pupilar = [float(pup[2])     for pup in pupila]
    
if len(pupilas) > len (pupilar): 
   pup_d = pupilas
else: 
   pup_d = pupilar
        
df = DataFrame({'onsets':onsets, 'pupilas':pup_d},index=onsets)

pupilas_f = blinkfix(df.pupilas)

pupilas_f = stats.zscore(pupilas_f)


plt.plot(onsets, pupilas_f)
    
plt.show()
