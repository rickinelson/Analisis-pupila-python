import psycopg2
import matplotlib.pyplot as plt
from matplotlib.pylab import *
import scipy
from scipy.signal import butter, filtfilt
from pandas import DataFrame
from bisect import bisect_left
import numpy as np
from scipy import stats

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

condicion1 = cursor.execute("SELECT cuenta, archivo, rt, condicion FROM datos WHERE congruency = 'congruent'  AND response = 'OK' AND condicion = 'V' AND d2 >= '0.3' AND d2 <= '0.5'and archivo not in ('dani_LV','inig_JV', 'dani_LT','inig_JT')")#azul
condicion1 = cursor.fetchall()

len(condicion1)

condicion2 = cursor.execute("SELECT cuenta, archivo, rt, condicion FROM datos WHERE congruency = 'incongruent' AND response = 'OK' AND condicion = 'V'  AND d2 >= '0.3' AND d2 <= '0.5' and archivo not in ('dani_LT','inig_JT', 'dani_LV','inig_JV')") #naranjo
condicion2 = cursor.fetchall()

len(condicion2)



def huachimingo (condicion, nombre_archivo):
    student = open (nombre_archivo, 'w+')    
    #sumaP= [0]*20000
    i = 0
    suj = []
    for trial in condicion:
        
        ARCHIVO = trial[1]
        print ARCHIVO, trial[0]
        	
        cursor.execute("SELECT tiempo, mensaje FROM mensajes WHERE archivo = '%s' AND mensaje LIKE 'trial:%d%%' ORDER BY tiempo" % (ARCHIVO, trial[0]))
        mensaje_actual = cursor.fetchall()[0]
        
        cursor.execute("SELECT tiempo, mensaje FROM mensajes WHERE archivo = '%s' AND mensaje LIKE 'trial:%d%%' ORDER BY tiempo" % (ARCHIVO, trial[0]+1))
        mensaje_siguiente = cursor.fetchall()
        	
        if len(mensaje_siguiente) == 0: continue
        	
        mensaje_siguiente = mensaje_siguiente[0]	
        
        cursor.execute("SELECT onset, pup_l, pup_r FROM pupila WHERE onset BETWEEN %d - 400 AND %d AND archivo = '%s' ORDER BY onset" % (mensaje_actual[0], mensaje_siguiente[0], ARCHIVO))
        pupila = cursor.fetchall()
        	
        if len(pupila) > 20000: continue
        i += 1
        	#tiempo = mensaje_actual[0] - 200
        	#tiempo2 = mensaje_actual[0]
        	
        	#cursor.execute("SELECT avg(pup_l),avg(pup_r) FROM pupila WHERE onset BETWEEN %d AND %d AND archivo = '%s'" % (tiempo, tiempo2, ARCHIVO))
        	#linea_base = cursor.fetchall()[0]
        	
        	#linea_base[0] # promedio pup_l
        	#linea_base[1] # promedio pup_r
        	#print linea_base[0]
        	#print linea_base[1]
        	
        cantidad_l_cero = len([pup[1] for pup in pupila if pup[1] == 0.0])
        cantidad_r_cero = len([pup[2] for pup in pupila if pup[2] == 0.0])
        	
        porcentaje_l_cero = 100*float(cantidad_l_cero)/len(pupila)
        porcentaje_r_cero = 100*float(cantidad_r_cero)/len(pupila)
        	
        print "%s: %d datos entre %d y %d. Lcero=%f%% Rcero=%f%%" % (mensaje_actual[1], len(pupila), mensaje_actual[0], mensaje_siguiente[0], porcentaje_l_cero, porcentaje_r_cero)
        	
    	#if porcentaje_l_cero > 90 or porcentaje_r_cero > 90:
    	#	continue
    	
    	
        tiempo_inicial = pupila[200+250][0]
        onsets = [pup[0]-tiempo_inicial for pup in pupila]
    	
        pupilas = [float(pup[1])     for pup in pupila] #revisar si es la mejor estrategia, ademas si promedias debiera ser despues del filtro
        
        pupilar = [float(pup[2])     for pup in pupila]
        
        if len(pupilas) > len (pupilar): 
            pup_d = pupilas
        else: 
            pup_d = pupilar
        
        df = DataFrame({'onsets':onsets, 'pupilas':pup_d},index=onsets)
    	
        
    	
        pupilas_f = blinkfix(df.pupilas)
    	
    	
        linea_base = mean(pupilas_f[150:250])
        	
    	
        pupilas_f = [x - linea_base for x in pupilas_f]
    	
        
    	#for y in range (len(pupilas)): #la creacion de lista de numeros desde 0 hata el final del parametro pupila 
    	#		sumaP[y] = sumaP[y] + pupilas_f[y]
        pupilas_f = pupilas_f[100:1500] # siempre dividir por dos y da el tiempo real, por que seleciona laposicion de la foto que dependen de la frecuencia de muestreo
        
        print(len(pupilas_f))
        if i == 1:
            sumaP = pupilas_f
        else:
            sumaP = np.vstack((sumaP,pupilas_f)) 
            
        suj.append(ARCHIVO)
        
        
    student.write(ARCHIVO +','+ ','.join(map(str, pupilas)) + "\n") # convierte la variable pupilas_f en string
    student.close()	
    	
    	
    	
    
    promediort = float(sum([trial[2]*1000 for trial in condicion]))/len(condicion)
    plt.axvline(0)  
    plt.axvline() 
    plt.axvline(promediort)
    onsetsp = range(-1000,1800,2)
    #plt.plot(onsetsp,[s/len(condicion) for s in sumaP], '-D', markevery =[bisect_left(onsetsp,promediort)])
    plt.plot(onsetsp, sumaP.mean(axis = 0), '-D', markevery =[bisect_left(onsetsp,promediort)])
    plt.xlabel('Tiempo en ms') 
    plt.suptitle('Dilatacion pupilar red control')
    plt.ylabel('Diferencia en la dilatacion pupilar') 
    plt.subplot()
    
    return (suj,sumaP)
    
    

sujs1, sumap1 = huachimingo(condicion1, 'congruentT')
sujs2, sumap2 = huachimingo(condicion2, 'incongruentT')


#if sum(unique(sujs1) == unique(sujs2)) == len(unique(sujs1)):
sujslist1 = unique(sujs1)
sujslist2 = unique(sujs2)
#else:
 #   print('cago la wea: las condiciones no tienen los mismos sujetos')

sujs_list1 = [[y == x for x in sujs1] for y in sujslist1]
sujs_mean1 = [mean(sumap1[x],axis = 0) for x in sujs_list1]

sujs_list2 = [[y == x for x in sujs2] for y in sujslist2]
sujs_mean2 = [mean(sumap2[x],axis = 0) for x in sujs_list2]


for x in range(len(sujs_mean1[0])): 
    g1 = [pups[x] for pups in sujs_mean1]
    g2 = [pups[x] for pups in sujs_mean2]      
    p = stats.ttest_rel(g1,g2)
 
	
    print (p)
	
    if p[1] < 0.1 and p[1] > 0.05: 
         plt.plot(2*x - 1000,0, marker= 'd', markersize = 2, color = 'yellow')
    elif p[1] < 0.05:
		plt.plot(2*x - 1000,0, marker= 'd', markersize = 5, color = 'green')
    else:
		plt.plot(2*x - 1000,0, marker= 'd', markersize = 5, color = 'red')

#log.write(pupilas_f)
#log.close()	
    


plt.plot([1,1,1], label="Congruent", color = 'blue')
plt.plot([1,1,1], label="incongruent", color = 'orange')
plt.plot([1,1,1], label= "respuesta", marker= 'd', markersize = 5, color = 'orange')
plt.plot([1,1,1], label="target on set")
plt.plot([0,0,0], label="target on set", marker = '|', markersize = 20, color = 'blue')
plt.legend(bbox_to_anchor=(0., 1.02, 1., 0.), loc= 3 ,ncol=4, borderaxespad=0)

    	 
plt.show()