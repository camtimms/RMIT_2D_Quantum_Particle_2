    # -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:10:04 2023

@author: Campbell Timms
"""

import numpy as np
import matplotlib.pyplot as plt

 # #Define values

N = 200
 # # N = Number of pixels/grid size
 
sigma = 0.05
 # #sigma value found in the Gaussian 
 
delta_t = 1
 # #change in time
 
k = -1e8
 # #quickness in space
 
m = 9.109537944498001e-31
 # #m-mass electron = 0.0005485803 (amu) * 1 amu =	1.660566E-27 kg	kg = 9.109537944498001e-31 kg
 
hbar = 1.054571817*10**-34 
 # #constant hbar: 1.054 571 817...x 10-34 J s or 6.582 119 569...  x 10-16 eV s 
 
t_total = 650
# the end time that delta_t will stop at 
 
pltfreq = 100
 # #How frequent through the time step delta_t will a time be plotted

potential = 1e-35
 # #Potential of the tophat middle, V
 
 # # Bohr radius (m) Griff eq.4.72
a = 0.529e-10
 
 # # Ground state of hydrogen eq.4.80 Griff
# psi100 = 1/(np.sqrt(np.pi*a**3))*np.exp(-r/a)

# defining the coordinates used
xyarrays = np.mgrid[:N,:N] 
y = xyarrays[0]
x = xyarrays[1]
# z = 
Y = (y - y[N-1,N-1]/2)
X = (x - x[N-1,N-1]/2)
# Z =

r = np.sqrt(X**2+Y**2) 

xyarrays = np.mgrid[:N,:N] 

q_x = xyarrays[0]
q_y = xyarrays[1]
#q_z = xyarrays[?]

q_X = (q_x - q_x[N-1,N-1]/2)
q_Y = (q_y - q_y[N-1,N-1]/2)
# q_Z = (q_z - q_z[N-1,N-1]/2)

shift = N//2
q_xshift = np.roll( q_X, int(-shift), 0)
q_yshift = np.roll( q_Y, int(-shift), 1)
# q_zshift = np.roll( q_Z, int(-shift), ?)

#Use the q-space arrays to make a complex array that stores the values of the free space equation 
qr2 = np.sqrt(q_xshift**2 + q_yshift**2)


def kenetic (wavefunction, delta_t, N = 200, sigma = 0.05, k = -1e8, m = 9.109537944498001e-31, hbar = 1.054571817*10**-34):
    
        Schro_time = np.exp((1j*hbar*delta_t/(2*m))*qr2**2)
            
        #Take the Fourier transform of the wavefunction at t=0.
        F_wavefunction = np.fft.fft2(wavefunction)
    
        # Multiply the Fourier transform of the wave function by the phase factor
        F_wavefunctiont = F_wavefunction*Schro_time
    
        # Take the inverse Fourier transform. 
        wavefunctiont = np.fft.ifft2(F_wavefunctiont)
        
        return wavefunctiont

# Wavefunction after time delta_t
wavefuntion_out = kenetic(np.exp(-r**2/(2*sigma**2)), delta_t)

# Input value of the tophat potential 

barrier_thickness_percentage = 0.05
width = int(N*barrier_thickness_percentage)
middle = N//2

tophat = np.zeros([N,N])
tophat[:,middle - width//2:middle + width//2 + 1] = potential

def potentialmultiplication(wf, V, delta_t, hbar = 1.054571817*10**-34):
    
        potentialt = np.exp(((-1j*delta_t)/hbar)*V)
    
        return wf*potentialt

wf = np.exp(-r**2/(2*sigma**2))
V = tophat

# Initialize the wave-function
N_constant = 1/np.sqrt(np.sum((wavefuntion_out)**2))
x0 = N//4
x_move = X+x0
int_wf = N_constant*np.exp((-(x_move**2+Y**2)/2*sigma**2)+1j*k*X)


#Create double slit in barrier 
slit_1_start = N//2-2*width
slit_1_end = N//2-width

slit_2_start = (N//2)+width
slit_2_end = (N//2)+2*width

barrier_thickness_percentage = 0.05
width = int(N*barrier_thickness_percentage)
tophat[slit_1_start:slit_1_end,:] = 0
tophat[slit_2_start:slit_2_end,:] = 0

#image of the potential
plt.imshow(tophat)
plt.title('Double slit')
plt.xlim(0,N)
plt.xlabel('Arbitary Spatial Units') 
plt.ylim(0,N)   
plt.ylabel('Arbitary Spatial Units')
plt.colorbar()
plt.show()

n = int(t_total/delta_t)

for i in range(n):
    
        wf_p = potentialmultiplication(int_wf, V, delta_t)
        wf_t = kenetic(wf_p,delta_t)
        int_wf = wf_t
        
        if i%pltfreq == 0:
            plt.imshow(abs(wf_t))
            plt.title('Absolute Value' + ' ' + 't=' + str(delta_t*i))
            plt.xlim(0,N)
            plt.xlabel('Arbitary Spatial Units') 
            plt.ylim(0,N)   
            plt.ylabel('Arbitary Spatial Units')
            plt.colorbar()
            plt.show()  