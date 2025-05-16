# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:22:45 2023

@author: Campbell Timms
"""
import numpy as np
import matplotlib.pyplot as plt

 # #Define values

N = 200
 # # N = Number of pixels/grid size
 
sigma = 0.1
 # #sigma value found in the Gaussian 
 
delta_t = 1
 # #change in time
 
k = -1e3
 # #quickness in space
 
m = (0.0005485803*1.660566*10**-27)
 # #m-mass electron = 0.0005485803 (amu) * 1 amu =	1.660566E-27	kg

hbar = 1.054571817*10**-34 
 # #constant hbar: 1.054 571 817...x 10-34 J s or 6.582 119 569...  x 10-16 eV s 
 
t_total = 10000
# the end time that delta_t will stop at 
 
pltfreq = 100
 # #How frequent through the time step delta_t will a time be plotted

potential = 1e100
 # #Potential of the tophat middle, V

 # # Sample Wavefuntion
 # # Gaussian = np.exp(-r**2/(2*sigma**2))

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


def kenetic (wavefunction, delta_t, N = 1000, L = 1, sigma = 0.1, k = 100, nplot = 10, m = (0.0005485803*1.660566*10**-27), hbar = 1.054571817*10**-34 ):
    Schro_time = np.exp((1j*hbar*delta_t/(2*m))*qr2**2)
        
    #Take the Fourier transform of the wavefunction at t=0.
    F_wavefunction = np.fft.fft2(wavefunction)

    # Multiply the Fourier transform of the wave function by the phase factor
    F_wavefunctiont = F_wavefunction*Schro_time

    # Take the inverse Fourier transform. 
    wavefunctiont = np.fft.ifft2(F_wavefunctiont)
        
    return wavefunctiont


# defining the coordinates used
xyarrays = np.mgrid[:N,:N] 
y = xyarrays[0]
x = xyarrays[1]
# z = 
Y = (y - y[N-1,N-1]/2)
X = (x - x[N-1,N-1]/2)
# Z =
r = np.sqrt(X**2+Y**2)  

# Wavefunction after time delta_t
wavefuntion_out = kenetic(np.exp(-r**2/(2*sigma**2)), delta_t)

# Plotting the absolute value of the wavefunction
# plt.imshow(abs(wavefuntion_out))
# plt.title('wavefuntion_out' + ' ' + 't=' + str(delta_t))

# plt.xlim(0,N)
# plt.xlabel('Arbitary Spatial Units') 

# plt.ylim(0,N)   
# plt.ylabel('Arbitary Spatial Units')
            
# plt.colorbar()
# plt.show()

# 2. Make a 2D array to store a potential. The values in each element of the array will be value of the potential. Set the potential to be constant non-zero value in some of the middle columns, and zero elsewhere (i.e. making a potential barrier or “top-hat” function). Display the potential array as an image.

# Input value of the tophat potential 

barrier_thickness_percentage = 0.05
width = int(N*barrier_thickness_percentage)
middle = N//2

tophat = np.zeros([N,N])
tophat[:,:] = potential

# plt.imshow(tophat)
# plt.title('Tophat array image')
# plt.colorbar()
# plt.show()

# 3. Write a function that multiplies the potential operator �!#$% ℏ + to an arbitrary wave function. V should be an input to the function so that you can reuse the function later with a different potential. Plot the �!#$% ℏ + function.


def potentialmultiplication(wf, V, delta_t, hbar = 1.054571817*10**-34):
    
    potentialt = np.exp(((-1j*delta_t)/hbar)*V)
    
    return wf*potentialt

wf = np.exp(-r**2/(2*sigma**2))
V = tophat

#plot of the absolute value
# plt.imshow(abs(potentialmultiplication(wf, V, 500)))
# plt.title('Potential multiplication (abs)')
# plt.colorbar()
# plt.show()

#plot of the phase (abs value doesn't change)
# plt.imshow(np.angle(potentialmultiplication(wf, V, 20)))
# plt.title('Potential multiplication (Phase)')
# plt.colorbar()
# plt.show()

# 4. Write a loop to solve equation (3) for n iterations. Initialize the wave-function as in Q8 from part 1,setting the gaussian width to be much smaller than your field-of-view and centring the gaussian on one edge of the x-axis. Predict the form of the wave function at several later times as it interacts with the potential barrier.

# Initialize the wave-function
N_constant = 1/np.sqrt(np.sum((wavefuntion_out)**2))
x0 = N//4
x_move = X+x0
int_wf = N_constant*np.exp((-(x_move**2+Y**2)/2*sigma**2)+1j*k*X)

# plt.imshow(abs(potentialmultiplication(int_wf, V, delta_t)))
# plt.title('Potential multiplication (int_wf)')
# plt.colorbar()
# plt.show()

#Evolve equation (3) for n iterations in time
#number of iterations n = number of times delta_t fits in the total time

n = int(t_total/delta_t)

# for i in range(n):
    
#     wf_p = potentialmultiplication(int_wf, V, delta_t)
#     wf_t = kenetic(wf_p,delta_t)
#     int_wf = wf_t
    
#     # if i%pltfreq == 0:
#         # plt.imshow(np.angle(wf_t))
#         # plt.title('Phase angle' + ' ' + 't=' + str(delta_t*i))
#         # plt.xlim(0,N)
#         # plt.xlabel('Arbitary Spatial Units') 
#         # plt.ylim(0,N)   
#         # plt.ylabel('Arbitary Spatial Units')
#         # plt.colorbar()
#         # plt.show()
    
#     if i%pltfreq == 0:
#         plt.imshow(abs(wf_t))
#         plt.title('Absolute Value' + ' ' + 't=' + str(delta_t*i))
#         plt.xlim(0,N)
#         plt.xlabel('Arbitary Spatial Units') 
#         plt.ylim(0,N)   
#         plt.ylabel('Arbitary Spatial Units')
#         plt.colorbar()
#         plt.show()    

# 5. Modify the potential from part 2 step 4 to put a gap in the potential (i.e. a slit). Repeat the simulation for part 2 step 4 with the new single-slit potential. 

barrier_thickness_percentage = 0.2
width = int(N*barrier_thickness_percentage)
tophat[middle - width//2:middle + width//2 + 1,:] = 0

# plt.imshow(tophat)
# plt.title('Tunnel')
# plt.xlim(0,N)
# plt.xlabel('Arbitary Spatial Units') 
# plt.ylim(0,N)   
# plt.ylabel('Arbitary Spatial Units')
# plt.colorbar()
# plt.show()

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

# 6. Make a potential of your own choosing and plot the results.





