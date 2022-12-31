# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 07:39:15 2022

@author: junse
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

f=open('test_result.txt', 'r')
L=f.readlines()
f.close()

data = np.zeros((5, 4))
for i in range(3, 8):
    tmp = L[i].split()
    for j in range(1, 5):
        data[i-3,j-1]= float(tmp[j])

plt.clf()
font = {'family' : 'normal',        
        'size'   : 25}

plt.rc('font', **font)

thick=1.5
fig,ax=plt.subplots(figsize = (10, 6.3))
plt.xlabel("Number of threads (streams)")
plt.ylabel('Process time per\nimage [ms]')



#scatter 그리기
dotsize =30

width =0.1





ax.tick_params(axis="y",direction="in", pad=10,length=8,width=thick)
ax.tick_params(axis="x",direction="in", pad=10,length=8,width=thick)
ax.tick_params(axis="y",which="minor",direction="in", pad=10,length=5,width=thick)
ax.tick_params(axis="x",which="minor",direction="in", pad=10,length=5,width=thick)
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(10))

ax.spines["top"].set_linewidth(thick)
ax.spines["bottom"].set_linewidth(thick)
ax.spines["left"].set_linewidth(thick)
ax.spines["right"].set_linewidth(thick)


ax.tick_params(axis="y",direction="in", pad=10)
ax.tick_params(axis="x",direction="in", pad=10)


plt.xticks(np.arange(0, 7,1))
plt.yticks(np.arange(20, 71,10))
plt.xlim(1, 6)
plt.ylim(20, 70)

image_sizes = [1, 2, 3, 4, 6]
ax.plot(image_sizes, data[:,0], 'o-', color = '#6868AC')
ax.plot(image_sizes, data[:,1], 'o-', color = '#E9435E')
ax.plot(image_sizes, data[:,2], 'o-', color = '#ECC371')
ax.plot(image_sizes, data[:,3], 'o-', color = '#85A1AC')

fig.savefig('cluster_points_time.png', bbox_inches='tight', dpi = 300)






