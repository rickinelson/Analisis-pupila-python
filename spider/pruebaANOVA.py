#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:53:26 2017

@author: Claudio
"""

import random

random.seed(1) #seed random number generator
cond_1 = [random.gauss(600,30) for x in range(30)] #condition 1 has a mean of 600 and standard deviation of 30
cond_2 = [random.gauss(650,30) for x in range(30)] #u=650 and sd=30
cond_3 = [random.gauss(600,30) for x in range(30)] #u=600 and sd=30

plt.bar(np.arange(1,4),[np.mean(cond_1),np.mean(cond_2),np.mean(cond_3)],align='center') #plot data
plt.xticks([1,2,3]);