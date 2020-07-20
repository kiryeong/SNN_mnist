# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 09:47:55 2020

@author: user
"""



from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

w_range = linspace(1,20,20)
duration = 1000*ms

for w in w_range:
    P = PoissonGroup(2001, rates=range(0,2001)*Hz)

    eqs = '''
    dv/dt=(I-v)/tau : 1
    I : 1
    tau : second
    '''
    
    G = NeuronGroup(2001, eqs, threshold='v>10', reset='v=0', refractory = 5*ms, method = 'exact')
    G.I = 0
    G.tau = 20*ms

    S = Synapses(P, G, on_pre='v += w')
    S.connect(j='i')


    M = SpikeMonitor(G, 'v', record = True)

    run(duration)

    plt.plot(range(0,2001)*w, M.count/duration, label="w=%d"%(w,))

plt.show()