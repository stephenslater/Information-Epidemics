# -*- coding: utf-8 -*-
"""
Created on Thu May  1 16:57:15 2014
Information Epidemics
AM50 Final Project
Harvard College, Spring 2014

Integrate system of difeqs and fit model to scraped Youtube view data of viral videos
Choose which model to use by uncommenting it and commenting out the other 2 models

@authors: Stephen Slater and Melissa Flores
"""

# Import necessary functions
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from scipy.integrate import odeint
import math

# Constants in the models (See paper for all values)
alpha=0.0005
beta=0.0093
gamma=0.0061
mu=0.0035
# alpha1=0.0001
# alpha2=0.01
# alpha3=0.043

# # Calculate derivatives S(t), I(t), R(t), E(t), Sh(t), Ih(t), Rh(t)
def deriv(x,t):
    
    # Model 1    
    # talk = beta * x[0] * x[1]
    # web = alpha * x[0] * x[1]
    # loss = gamma * x[1]
    # return np.array([-talk - web, talk + web - loss, loss])

    # Model 2
    # dS = -x[0] * (beta * x[1] + alpha1 * x[1] + alpha2 * x[4])
    # dI = x[0] * (beta * x[1] + alpha1 * x[1] + alpha2 * x[4]) - gamma * x[1]
    # dR = gamma * x[1]
    # dS(H) = -x[3] * (beta * x[1] + alpha2 * x[1] + alpha3 * x[4])
    # dI(H) = x[3] * (beta * x[1] + alpha2 * x[1] + alpha3 * x[4]) - gamma * x[4]
    # dR(H) = gamma * x[4]
    # return np.array([dS, dI, dR, dS(H), dI(H), dR(H)])

    # Model 3
    talk = beta * x[0] * x[2]
    web = alpha * x[0] * x[2]
    infect= mu * x[1]
    loss = x[2] * (math.e ** (-gamma * t))
    return np.array([-talk - web, talk + web - infect, infect - loss, loss])


# See paper for time length and initial values of S, I, R, and E
time = np.linspace(0, 1500, 8000)
xinit = np.array([360, 22, 3, 9])

# Solve ODE using the "odeint" library in SciPy
x = odeint(deriv, xinit, time)

# Read in image to array
youtube = io.imread('gangnamdailyviews.png', as_grey=True)

# List of pixel height of datapoints in vertical image scan
raw = []

# How tall and how wide the screenshot of the views/day graph is
rows = youtube.shape[0]
columns = youtube.shape[1]

# For every column in every row, look at that pixel
for i in range(columns):
    for j in range(rows):
        pixel = youtube[j][i]
        
        # Store the colored pixel height (with cutoff intensity 0.80)
        if (pixel < 0.80):
            raw.append(rows-j)            
            break

# List of transformed values of views/day; not used for inaccuracy
# transform = []
# for k in raw:
#    multiply = 30000*raw[k] # dailyview.png 46012*raw[k] # youtubedata.png 14330*raw[k]
#    transform.append(multiply)

# Plot model and Youtube
plt.figure()

# Plot Model 1 with Youtube graph
# p1,=plt.plot(time,x[:,1])
# p2,=plt.plot(raw)
# plt.legend([p1,p2],["Model","Raw"])
# plt.title('Gangnam Style Model1')
# plt.xlabel('Time')
# plt.ylabel('Population I (views/day)')

# Plot Model 2 with Youtube graph
# p1,=plt.plot(time,x[:,1]+x[:,4])
# p2,=plt.plot(time,x[:,3])
# p3,=plt.plot(raw)
# plt.legend([p1,p3],["Model","Raw"])
# plt.title('Gangnam Style Model2')
# plt.xlabel('Time')
# plt.ylabel('Population I + Population Ih (views/day)')

# Plot Model 3 with Youtube graph
p1,=plt.plot(time, x[:,1] + x[:,2])
p3,=plt.plot(raw)
plt.legend([p1, p3],["Model", "Raw"])
plt.title('Gangnam Style Model3')
plt.xlabel('Time')
plt.ylabel('Population I + Population E (views/day)')

plt.show()

# Plot the transformation of Youtube graph
# plt.figure()
# p0,=plt.plot(transform)
# plt.legend([p0],["Transform"])
# plt.xlabel('time')
# plt.ylabel('Views/day')
# plt.show()  
