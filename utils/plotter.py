#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:11:06 2025

@author: ejg2736
"""

import matplotlib.pyplot as plt

def plot_alongtrack(alongtrack,h_ph,combined_class_ph,title):
    plt.figure()
    plt.plot(alongtrack[combined_class_ph == 0][::100],h_ph[combined_class_ph == 0][::100],'.',color=[0.8,0.8,0.8],label='Unclassified')
    plt.plot(alongtrack[combined_class_ph == 3],h_ph[combined_class_ph == 3],'.',color=[0.20392157, 0.70196078, 0.20392157],label='Top of Canopy (3)')
    plt.plot(alongtrack[combined_class_ph == 2],h_ph[combined_class_ph == 2],'.',color=[0.12156863, 0.41960784, 0.12156863],label='Canopy (2)')
    plt.plot(alongtrack[combined_class_ph == 1],h_ph[combined_class_ph == 1],'.',color=[0.69803922, 0.44313725, 0.23921569],label='Terrain (1)')
    plt.plot(alongtrack[combined_class_ph == 40],h_ph[combined_class_ph == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='Subaqueous Terrain (40)')
    plt.plot(alongtrack[combined_class_ph == 41],h_ph[combined_class_ph == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='Water Surface (41)')
    plt.xlabel('Alongtrack (m)')
    plt.ylabel('Ellipsoid Height (m)')
    plt.legend()
    plt.title(title)
    plt.show()