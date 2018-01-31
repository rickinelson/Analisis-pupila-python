import psycopg2
import matplotlib.pyplot as plt
from matplotlib.pylab import *
import scipy
from scipy.signal import butter, filtfilt
from pandas import DataFrame
from bisect import bisect_left
import numpy as np

conn = psycopg2.connect("host='localhost' dbname='datos' user='postgres' password='123456789'")
cursor = conn.cursor()


def archivo():
	txt=open('%OKNOK', 'w+')
	txt.close()
	
def porcentaje(con, x, y, z, n, o, t, pn, po):
	txt=open('%OKNOK', 'a')
	txt.write(', '.join(map(str,(con, n, x,))) + "\n")
	txt.write(', '.join(map(str,(con, o, y,)))+ "\n")
	txt.write(', '.join(map(str,(con, t, z,)))+ "\n")
	txt.write(', '.join(map(str,(con, pn, round((x*100.0)/z, 2))))+ "\n")
	txt.write(', '.join(map(str,(con, po, round((y*100.0)/z, 2))))+ "\n")
	txt.close()


def condicion(con):

	condicion1 = cursor.execute("SELECT response FROM datos WHERE response = 'NOK'  AND condicion = '%s'" %con)
	condicion1 = cursor.fetchall()
	condicion2 = cursor.execute("SELECT response FROM datos WHERE response = 'OK' AND  condicion = '%s'" %con)
	condicion2 = cursor.fetchall()
	
	x = len(condicion1)
	y = len(condicion2)
	z = x + y
	n = 'NOK'
	o = 'OK'
	t = 'total'
	pn = 'porcentaje NOK'
	po = 'porcentaje OK'
	p = '%'

	
	porcentaje(con, x, y, z, n, o, t, pn, po)

print condicion('C')
print condicion('T')
print condicion('V')
print condicion('H')


