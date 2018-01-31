#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 22:23:11 2017

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



def inicio (query_mensaje, query_pupila):
    condicion = 'C'
    
    cursor.execute(query_mensaje % (condicion))
    inicio_estimulacion = cursor.fetchall()
    
    i = 0
    suj = []
    for x in inicio_estimulacion :
        
        print x
        x[0]
        x[2]
        cursor.execute(query_pupila % (x[0], x[0], condicion, x[2]))
        pupila = cursor.fetchall()
    
        i += 1
        
        cantidad_l_cero = len([pup[1] for pup in pupila if pup[1] == 0.0 or pup[1] is None])
        cantidad_r_cero = len([pup[2] for pup in pupila if pup[2] == 0.0 or pup[2] is None])
            	
        porcentaje_l_cero = 100*float(cantidad_l_cero)/len(pupila)
        porcentaje_r_cero = 100*float(cantidad_r_cero)/len(pupila)
    
        onsets = [pup[0] for pup in pupila]
    
        pupilas = [float(pup[1])     for pup in pupila if pup[1] is not None] #revisar si es la mejor estrategia, ademas si promedias debiera ser despues del filtro
            
        pupilar = [float(pup[2])     for pup in pupila if pup[2] is not None]
            
        if len(pupilas) > len (pupilar): 
            pup_d = pupilas
        else: 
            pup_d = pupilar
            
        df = DataFrame({'onsets':onsets, 'pupilas':pup_d},index=onsets)
    
        pupilas_f = blinkfix(df.pupilas)
        
        pupilas_f = pupilas_f[0:29990]
    
        print len(pupilas_f)
    
        pupilas_f = stats.zscore(pupilas_f)
    
        if i == 1:
           sumaP = pupilas_f
        else:
           sumaP = np.vstack((sumaP,pupilas_f)) 
           
        suj.append(x[2])
        
        
        
    return (suj,sumaP)

condicion = 'T'

#query_mensaje = "SELECT tiempo, mensaje, archivo from mensajes where mensaje like 'trial:1 trial: 1' AND condicion = '%s' ORDER BY tiempo" #final naranjo
#query_pupila = "SELECT onset, pup_l, pup_r FROM pupila WHERE onset BETWEEN %d - 45000  AND %d  AND condicion = '%s' AND ARCHIVO = '%s' ORDER BY onset" #final  

query_mensaje2 = "SELECT tiempo, mensaje, archivo from mensajes where mensaje like '%%trial:0 inicio estimulacion%%' AND condicion = '%s' ORDER BY tiempo" #inicio
query_pupila2 = "SELECT onset, pup_l, pup_r FROM pupila WHERE onset BETWEEN %d  AND %d + 60000 AND condicion = '%s' AND ARCHIVO = '%s' ORDER BY onset" #inicio
 
#sujs1, sumaP = inicio(query_mensaje, query_pupila)
sujs2, sumaP2 = inicio(query_mensaje2, query_pupila2) #llamada de funcion


#sujslist1 = unique(sujs1)
sujslist2 = unique(sujs2)

#sujs_list1 = [[y == x for x in sujs1] for y in sujslist1]
#sujs_mean1 = [mean(sumaP[x],axis = 0) for x in sujs_list1]

sujs_list2 = [[y == x for x in sujs2] for y in sujslist2]
sujs_mean2 = [mean(sumaP2[x],axis = 0) for x in sujs_list2]




selectiempo = []

#for x in range(len(sujs_mean1[0])): 
    #g1 = [pups[x] for pups in sujs_mean1]
    #g2 = [pups[x] for pups in sujs_mean2] 
    #print g1
    #print g2     
    #p = stats.ttest_rel(g1,g2)
    
    #print (p)
    
   # if p[1] < 0.05:
        #plt.plot(2*x,0, marker= 'd', markersize = 2, color = 'green')
        #selectiempo.append(x)
    #else:
        #plt.plot(2*x,0, marker= 'd', markersize = 2, color = 'red')

onsetsp = range(0,29990*2,2)
print onsetsp
#plt.plot(onsetsp, sumaP.mean(axis = 0), color = 'orange') #final
plt.plot(onsetsp, sumaP2.mean(axis = 0), color = 'blue') #inicio
plt.suptitle('Estimulacion tactil',fontsize=25) 
plt.ylabel('Z score',fontsize=25) 
plt.xlabel('Tiempo en ms', fontsize=25) 
plt.plot([1,1,1], label="Inicio 5 a 45 segundos",markersize = 8, color = 'blue')
plt.plot([1,1,1], label="final 20 a 60 segundos", markersize = 8,color = 'orange')
plt.plot([1,1,1], label= "p < 0,05", marker= '_', markersize = 8, color = 'green')
plt.plot([1,1,1], label= "p > 0,05", marker= '_', markersize = 8, color = 'red')
plt.legend(bbox_to_anchor=(0., 1.02, 1., 0.), loc= 3 ,ncol=4, borderaxespad=0, fontsize = 'xx-large')



plt.show()