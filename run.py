
#
# Some notes on implementation for better or worse
# 
# Original program written by Sigrid Bratsberg, credit where credit is due! 
# 
# Switching from printing to a single file to making multiple out outs for a given
# set of input values is enabled by decommenting the newvar lines
# 
# TODO
# Someone should move all the parameters to the main program
#

#import pandas
import os
import sys

sys.path.append('/home/elf/elfdata/python')
os.chdir('/home/elf/elfdata/python')

from multiprocessing import Pool
import models 
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
#from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
from copy import deepcopy
#import seaborn as sns
#import pygraphviz as pgv
from statistics import stdev, mean
import imageio
import networkx as nx
from scipy.stats import truncnorm
from itertools import repeat
import time
import multiprocessing
from pathlib import Path
import dill



#random.seed(1574705741)    ## if you need to set a specific random seed put it here
#np.random.seed(1574705741)

if __name__ ==  '__main__': 

    #Constants and Variables
    numberOfSimulations = 80
    numberOfProcessors = 16 # CPUs to use for parallelization

    start = time.time()
    pool=Pool( processes = numberOfProcessors) #initializing pool
    
    # ----------PATH TO SAVE FIGURES AND DATA-----------

    pathFig = '/home/elf/elfdata/output'
    pathData = '/home/elf/elfdata/output'
    
    modelargs=models.getargs()  # requires models.py to be imported
    runs = 2   ## has to be even for multiple runs also n is actually n-1 because I'm lazy

    ### comment out all below for single run
    
    ### linear grid 
    grid = [1]

    ### log grid, only valid on range [-1,1] atm

    #var = "newPoliticalClimate"

    #steps = int(runs/2)
    #start = modelargs[var]
    #endup = 0.10
    #enddw = -0.45
    #logendup = np.log(endup+(1.0-start))
    #logenddw = np.log(enddw+(1.0-start))
    #stepup = logendup / steps
    #stepdw = logenddw / steps

    #gridup = np.array([])
    #griddw = np.array([])

    #for k in range (steps) :
    #    pt = np.exp(stepup*k)
    #    gridup = np.append(gridup,pt)
    #
    #for k in range (steps) :
    #    pt = np.exp(stepdw*k)
    #    griddw = np.append(griddw,pt)

    #gridup = gridup - (1.0-start)
    #griddw = griddw - (1.0-start)

    #griddw = griddw[1:]
    #griddw = np.flip(griddw)

    #grid = np.append(griddw,gridup)

    #print (grid)

    #for run in range(runs-1) :
    for run in grid :
        print("Started iteration: ", run)

        argList = []
        
        ## You can specify simulation parameters here. If they are not set here, they will default to some values set in models.py
        argList.append({"influencers": 0})
        #argList.append({"influencers" : 0, "type" : "cl"})
       
        #print (argList)

        for i in range(len(argList)):
            sim = pool.starmap(models.simulate, zip(range(numberOfSimulations), repeat(argList[i])))

            fname = './biased_linkifsame_2networksteps{}.csv'.format(run)
            models.saveavgdata(sim, fname)

    end = time.time()
    mins = (end - start) / 60
    sec = (end - start) % 60
    print(f'Runtime was complete: {mins:5.0f} mins {sec}s\n')
