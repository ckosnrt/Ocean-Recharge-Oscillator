# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 17:20:39 2024

@author: Sinurat
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
def euler_forw(T_e_init, h_w_init, time, nt, dt, mu_0, xi_2, e_n, mu_ann, tau, tau_cor, f_ann, f_ran, r=0.25, c=1, alpha=0.125, b_o=2.5, gamma=0.75):
    """
    Integrates the system of differential equations using the forward Euler method.

    Parameters
    ----------
    T_e_init : float. Initial temperature value
    h_w_init : float. Initial ocean depth value
    time : ndarray. Array of time points
    nt : int. Number of time points
    dt : float. Time step size
    mu_0 : float. Initial coupling coefficient.
    xi_2 : float. Random heating added the system
    e_n : float. Varies the degree of nonlinearity
    mu_ann : float. Annual coupling coefficient
    tau : float. Period of the annual cycle
    tau_cor : float. Correlation time scale
    f_ann : float. Amplitude of the annual cycle forcing
    f_ran : float. Amplitude of random fluctuation forcing
    r : float. Damping of the upper ocean heat content. The default is 0.25
    c : float. Damping rate of SST anomalies. The default is 1.
    alpha : float. easterly wind stress. The default is 0.125.
    b_o : float. high-end value of the coupling parameter. The default is 2.5.
    gamma : float. feedback of thermocline gradient. The default is 0.75.

    Returns
    -------
    ndarray. Array of temperature values
    ndarray. Array of ocean depth
    
    """
    b = b_o * mu_0
    R = gamma * b - c
    
    T_e = np.zeros(nt)
    h_w = np.zeros(nt)
    
    T_e[0] = T_e_init
    h_w[0] = h_w_init

    for i in range(nt-1):
        time = dt*i
        xi_1 = wind(time, dt, f_ann, f_ran, tau, tau_cor)
        
        h_w[i+1] = h_w[i] + dt * ((-r * h_w[i]) - (alpha * b * T_e[i]) - (alpha * xi_1))
        T_e[i+1] = T_e[i] + dt * ((R * T_e[i]) + (gamma * h_w[i]) - (e_n * (h_w[i] + b * T_e[i])**3) + (gamma * xi_1) + xi_2)
    
    return T_e*7.5, h_w*15


#%%
def dfdx(T_e, h_w, time, mu_0, xi_2, e_n, mu_ann, tau, tau_cor, dt, f_ann, f_ran, r=0.25, c=1, alpha=0.125, b_o=2.5, gamma=0.75):
    """
    Computes the derivative of the system of differential equations

    Parameters
    ----------
    T_e : float. Current temperature value
    h_w : float. Current ocean depth value
    time : float. Current time value
    mu_0 : float. Initial coupling coefficient.
    xi_2 : float. Random heating added the system
    e_n : float. Varies the degree of nonlinearity
    mu_ann : float. Annual coupling coefficient
    tau : float. Period of the annual cycle
    tau_cor : float. Correlation time scale
    dt : float. Time step size
    f_ann : float. Amplitude of the annual cycle forcing
    f_ran : float. Amplitude of random fluctuation forcing
    r : float. Damping of the upper ocean heat content. The default is 0.25
    c : float. Damping rate of SST anomalies. The default is 1.
    alpha : float. easterly wind stress. The default is 0.125.
    b_o : float. high-end value of the coupling parameter. The default is 2.5.
    gamma : float. feedback of thermocline gradient. The default is 0.75.

    Returns
    -------
    New_T_e : float. Derivative of temperature with respect to time
    New_h_w : float. Derivatice of ocean depth with respect to time

    """
    mu = func_mu(time, mu_0, mu_ann, tau) 
    xi_1 = wind(time, dt, f_ann, f_ran, tau, tau_cor)
    b = b_o * mu
    R = gamma * b - c
    
    New_T_e = (R * T_e) + (gamma * h_w) - (e_n * (h_w + b * T_e)**3) + (gamma * xi_1) + xi_2
    New_h_w = -(r * h_w) - (alpha * b * T_e) - (alpha * xi_1)
    
    return New_T_e, New_h_w

#%%
def func_mu(time, mu_0, mu_ann, tau):
    """
    Calculate the time-dependent coupling coefficient 'mu' with an annual cycle variation.

    Parameters
    ----------
    time : float. Current time value
    mu_0 : float. Initial coupling coefficient.
    mu_ann : float. Annual coupling coefficient
    tau : float. Period of the annual cycle
    
    Returns
    -------
    mu : float. coupling coefficient mu at the given time

    """
    mu = mu_0 * (1 + mu_ann * np.cos((2 * np.pi * time / tau) - (5 * np.pi / 6)))
    return mu

#%%
def wind(time, dt, f_ann, f_ran, tau, tau_cor):
    """
    Generate a wind force term combining both annual periodic and random fluctuation components.

    Parameters
    ----------
    time : float. Current time value
    dt : float. Time step size
    f_ann : float. Amplitude of the annual cycle forcing
    f_ran : float. Amplitude of random fluctuation forcing
    tau : float. Period of the annual cycle
    tau_cor : float. Correlation time scale
    
    Returns
    -------
    xi_1 : float. wind force xi_1 at the given time

    """
    W = np.random.uniform(-1, 1)
    xi_1 = f_ann * np.cos(2 * np.pi * time / tau) + f_ran * W * tau_cor / dt
    return xi_1

#%%
def RK4(T_e_init, h_w_init, nt, dt, mu_0, xi_2, e_n, mu_ann, tau, tau_cor, f_ann, f_ran):
    """
    Perform the 4th-order Runge-Kutta integration

    Parameters
    ----------
    T_e_init : float. Initial temperature value
    h_w_init : float. Initial ocean depth value
    nt : int. Number of time points
    dt : float. Time step size
    mu_0 : float. Initial coupling coefficient.
    xi_2 : float. Random heating added the system
    e_n : float. Varies the degree of nonlinearity
    mu_ann : float. Annual coupling coefficient
    tau : float. Period of the annual cycle
    tau_cor : float. Correlation time scale
    f_ann : float. Amplitude of the annual cycle forcing
    f_ran : float. Amplitude of random fluctuation forcing
    
    Returns
    -------
    ndarray. Array of temperature values
    ndarray. Array of ocean depth
    """
    T_e = np.zeros(nt)
    h_w = np.zeros(nt)
    
    T_e[0] = T_e_init
    h_w[0] = h_w_init

    for i in range(nt-1):
        time = dt*i
        k1, l1 = dfdx(T_e[i], h_w[i], time, mu_0, xi_2, e_n, mu_ann, tau, tau_cor, dt, f_ann, f_ran)
        k2, l2 = dfdx(T_e[i]+0.5*k1*dt, h_w[i]+0.5*l1*dt, time+dt/2, mu_0, xi_2, e_n, mu_ann, tau, tau_cor, dt, f_ann, f_ran)
        k3, l3 = dfdx(T_e[i]+0.5*k2*dt, h_w[i]+0.5*l2*dt, time+dt/2, mu_0, xi_2, e_n, mu_ann, tau, tau_cor, dt, f_ann, f_ran)
        k4, l4 = dfdx(T_e[i]+k3*dt, h_w[i]+l3*dt, time+dt, mu_0, xi_2, e_n, mu_ann, tau, tau_cor, dt, f_ann, f_ran)
        
        T_e[i+1] = T_e[i] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        h_w[i+1] = h_w[i] + (dt/6) * (l1 + 2*l2 + 2*l3 + l4)

    return T_e*7.5, h_w*15


#%%
def plot(time, T_e, h_w, title):
    """
    Create a plot of the temperature and ocean depth

    Parameters
    ----------
    time : ndarray. Array of time points
    T_e : float. Current temperature value
    h_w : float. Current ocean depth value
    title : str. The title of the plot

    Returns
    -------
    displays a plot

    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sc = axes[0].scatter(T_e, h_w, c=h_w, cmap='summer', edgecolor='none', s=10)
    axes[0].set_xlabel('$T_e$, $^oC$')
    axes[0].set_ylabel('$h_w$, 10m')
    axes[0].set_title('$T_e$ Vs $h_w$')
    axes[0].grid(True)
    axes[0].plot(T_e[0], h_w[0], 'or', label='start')
    fig.colorbar(sc, ax=axes[0])
    
    axes[1].plot(time, T_e, 'g--', label='$T_e$')
    axes[1].plot(time, h_w, 'r--', label='$h_w$')
    axes[1].set_xlabel('time, months')
    axes[1].set_ylabel('$h_w$ (10m), $T_e$ ($^oC$)')
    axes[1].set_title('Time Series')
    axes[1].legend()
    axes[1].grid(True)
    
    fig.suptitle(title)
    plt.show()   
  
    
#%%
def task(nop, dt=1/41, mu_0=2/3, xi_2=0, e_n=0, mu_ann=0, tau=6, tau_cor=1, f_ann=0, f_ran=0, euler=False):
    """
    Perform a simulation for a given number of oscillation periods

    Parameters
    ----------
    nop : int. Number of oscillation periods
    dt : float. Time step size.  The default is 1.
    mu_0 : float. Initial coupling coefficient. The default is 2/3.
    xi_2 : float. Random heating added the system. The default is 0.
    e_n : float. Varies the degree of nonlinearity. The default is 0.
    mu_ann : float. Annual coupling coefficient. The default is 0.
    tau : float. Period of the annual cycle. The default is 365.
    tau_cor : float. Correlation time scale. The default is 1.
    f_ann : float. Amplitude of the annual cycle forcing. The default is 0.
    f_ran : float. Amplitude of random fluctuation forcing. The default is 0.
    euler : function. Euler forward method. The default is False.

    Returns
    -------
    plot a simulation.

    """
    T_e_init = 1.125 / 7.5
    h_w_init = 0 / 15
    omega_c = np.sqrt(3 / 32)
    period = 2 * np.pi / omega_c
    nt =int(nop*period/dt)    
    time = np.arange(0, nt*dt, dt)
    
    if euler==True:
        euler_T_e, euler_h_w = euler_forw(T_e_init, h_w_init, time, nt, dt, mu_0, xi_2, e_n, mu_ann, tau, tau_cor, f_ann, f_ran)
        plot(time, euler_T_e, euler_h_w, f"Euler Forward for {nop} period(s) with $\mu_0$ = {mu_0:.2}")
    else:
        RK4A_T_e, RK4A_h_w = RK4(T_e_init, h_w_init, nt, dt, mu_0, xi_2, e_n, mu_ann, tau, tau_cor, f_ann, f_ran)
        plot(time, RK4A_T_e, RK4A_h_w, f"Runge-Kutta 4th for {nop} period(s) with $\mu_0$ = {mu_0:.2}")    
    
        
#%%
def taskG(nop, dt=1/60, mu_0=0.75, xi_2=0, e_n=0.1, mu_ann=0.2, tau=6, tau_cor=1/60, f_ann=0.02, f_ran=0.2):
    """
    Perform a simulation of ensemble model

    Parameters
    ----------
    nop : int. Number of oscillation periods
    dt : float. Time step size. The default is 1.
    mu_0 : float. Initial coupling coefficient. The default is 2/3.
    xi_2 : float. Random heating added the system. The default is 0.
    e_n : float. Varies the degree of nonlinearity. The default is 0.
    mu_ann : float. Annual coupling coefficient. The default is 0.
    tau : float. Period of the annual cycle. The default is 365.
    tau_cor : float. Correlation time scale. The default is 1.
    f_ann : float. Amplitude of the annual cycle forcing. The default is 0.
    f_ran : float. Amplitude of random fluctuation forcing. The default is 0.

    Returns
    -------
    plot a ensemble simulation.

    """
    T_e_init = 1.125 / 7.5
    h_w_init = 0 / 15
    omega_c = np.sqrt(3 / 32)
    period = 2 * np.pi / omega_c
    nt =int(nop*period/dt)    
    esem_size = 50
    time = dt * np.linspace(0, nt-1, nt)
    dT = 0.02
    dh = 0.02
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 4))
    
    for i in range(esem_size):
        T_e_random = T_e_init + np.random.uniform(-dT, dT)
        h_w_random = h_w_init + np.random.uniform(-dh, dh)
        RK4A_T_e_random, RK4A_h_w_random = RK4(T_e_random, h_w_random, nt, dt, mu_0, xi_2, e_n, mu_ann, tau, tau_cor, f_ann, f_ran)
        
        axes[0].plot(time, RK4A_T_e_random, alpha=0.2, color='blue')
        axes[1].plot(time, RK4A_h_w_random, alpha=0.2, color='red')
    
    axes[0].set_xlabel('time, months')
    axes[0].set_ylabel('$T_e$, $^oC$')
    axes[0].set_title('Time Series of $T_E$')
    axes[0].grid(True) 

    axes[1].set_xlabel('time, months')
    axes[1].set_ylabel('$h_w$, 10m')
    axes[1].set_title('Time Series of $h_w$')
    axes[1].grid(True) 
    return


