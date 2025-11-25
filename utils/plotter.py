#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:11:06 2025

@author: ejg2736
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_alongtrack(alongtrack,h_ph,combined_class_ph,title):
    plt.figure()
    plt.plot(alongtrack[combined_class_ph == 0][::100],h_ph[combined_class_ph == 0][::100],'.',color=[0.8,0.8,0.8],label='Unclassified')
    plt.plot(alongtrack[combined_class_ph == 3],h_ph[combined_class_ph == 3],'.',color=[0.20392157, 0.70196078, 0.20392157],label='Top of Canopy (3)')
    plt.plot(alongtrack[combined_class_ph == 2],h_ph[combined_class_ph == 2],'.',color=[0.12156863, 0.41960784, 0.12156863],label='Canopy (2)')
    plt.plot(alongtrack[combined_class_ph == 1],h_ph[combined_class_ph == 1],'.',color=[0.69803922, 0.44313725, 0.23921569],label='Topography (1)')
    plt.plot(alongtrack[combined_class_ph == 41],h_ph[combined_class_ph == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='Water Surface (41)')
    plt.plot(alongtrack[combined_class_ph == 40],h_ph[combined_class_ph == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='Bathymetry (40)')
    plt.xlabel('Alongtrack (m)')
    plt.ylabel('Orthometric Height (m)')
    plt.legend(markerscale=3)
    plt.title(title)
    plt.show()
    
def plot_alongtrack_atl08(alongtrack,h_ph,atl08_class,title):
    plt.figure()
    plt.plot(alongtrack[atl08_class == 0],h_ph[atl08_class == 0],'.',color=[0.8,0.8,0.8],label='Unclassified')
    plt.plot(alongtrack[atl08_class == 3],h_ph[atl08_class == 3],'.',color=[0.20392157, 0.70196078, 0.20392157],label='Top of Canopy (3)')
    plt.plot(alongtrack[atl08_class == 2],h_ph[atl08_class == 2],'.',color=[0.12156863, 0.41960784, 0.12156863],label='Canopy (2)')
    plt.plot(alongtrack[atl08_class == 1],h_ph[atl08_class == 1],'.',color=[0.69803922, 0.44313725, 0.23921569],label='Topography (1)')
    plt.xlabel('Alongtrack (m)')
    plt.ylabel('Orthometric Height (m)')
    plt.legend(markerscale=3)
    plt.title(title)
    plt.show()
    
def plot_alongtrack_atl24(alongtrack,h_ph,atl24_class,title):
    plt.figure()
    plt.plot(alongtrack[atl24_class == 0],h_ph[atl24_class == 0],'.',color=[0.8,0.8,0.8],label='Unclassified')
    plt.plot(alongtrack[atl24_class == 41],h_ph[atl24_class == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='Water Surface (41)')
    plt.plot(alongtrack[atl24_class == 40],h_ph[atl24_class == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='Bathymetry (40)')
    plt.xlabel('Alongtrack (m)')
    plt.ylabel('Orthometric Height (m)')
    plt.legend(markerscale=3)
    plt.title(title)
    plt.show()
    
    
def plot_alongtrack_atl08_atl24(alongtrack,h_ph,atl08_class,atl24_class,title):
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    axes[0].plot(alongtrack[atl08_class == 0],h_ph[atl08_class == 0],'.',color=[0.8,0.8,0.8],label='Unclassified')
    axes[0].plot(alongtrack[atl08_class == 3],h_ph[atl08_class == 3],'.',color=[0.20392157, 0.70196078, 0.20392157],label='Top of Canopy (3)')
    axes[0].plot(alongtrack[atl08_class == 2],h_ph[atl08_class == 2],'.',color=[0.12156863, 0.41960784, 0.12156863],label='Canopy (2)')
    axes[0].plot(alongtrack[atl08_class == 1],h_ph[atl08_class == 1],'.',color=[0.69803922, 0.44313725, 0.23921569],label='Topography (1)')
    axes[0].set_xlabel('Alongtrack (m)')
    axes[0].set_ylabel('Orthometric Height (m)')
    axes[0].legend(markerscale=3)
    axes[0].set_title('ATL08 Classifications')
    
    axes[1].plot(alongtrack[atl24_class == 0],h_ph[atl24_class == 0],'.',color=[0.8,0.8,0.8],label='Unclassified')
    axes[1].plot(alongtrack[atl24_class == 41],h_ph[atl24_class == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='Water Surface (41)')
    axes[1].plot(alongtrack[atl24_class == 40],h_ph[atl24_class == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='Bathymetry (40)')
    axes[1].set_xlabel('Alongtrack (m)')
    axes[1].set_ylabel('Orthometric Height (m)')
    axes[1].legend(markerscale=3)
    axes[1].set_title('ATL24 Classifications')
    
    fig.suptitle(title, fontsize=16)
    plt.show()
    
    
def plot_alongtrack_contested_classes(alongtrack,h_ph,contested_class,title):
    plt.figure()
    plt.plot(alongtrack[contested_class == 0],h_ph[contested_class == 0],'.',color=[0.8,0.8,0.8],label='Unclassified Photon')
    plt.plot(alongtrack[contested_class == 1],h_ph[contested_class == 1],'.',color=[0.12156863, 0.81960784, 0.12156863],label='ATL08 Photon')
    plt.plot(alongtrack[contested_class == 2],h_ph[contested_class == 2],'.',color=[0.        , 0.61568627, 0.96862745],label='ATL24 Photon')
    plt.plot(alongtrack[contested_class == 3],h_ph[contested_class == 3],'.',color=[1,0,0],label='Contested Photon')
    plt.xlabel('Alongtrack (m)')
    plt.ylabel('Orthometric Height (m)')
    plt.legend(markerscale=3)
    plt.title(title)
    plt.show()
    
    
def plot_alongtrack_atl08_atl24_contested(alongtrack,h_ph,atl08_class,atl24_class,contested_class,title):
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    axes[0].plot(alongtrack[atl08_class == 0],h_ph[atl08_class == 0],'.',color=[0.8,0.8,0.8],label='Unclassified')
    axes[0].plot(alongtrack[atl08_class == 3],h_ph[atl08_class == 3],'.',color=[0.20392157, 0.70196078, 0.20392157],label='Top of Canopy (3)')
    axes[0].plot(alongtrack[atl08_class == 2],h_ph[atl08_class == 2],'.',color=[0.12156863, 0.41960784, 0.12156863],label='Canopy (2)')
    axes[0].plot(alongtrack[atl08_class == 1],h_ph[atl08_class == 1],'.',color=[0.69803922, 0.44313725, 0.23921569],label='Topography (1)')
    # axes[0].set_xlabel('Alongtrack (m)')
    axes[0].set_ylabel('Orthometric Height (m)')
    axes[0].legend(markerscale=3)
    axes[0].set_title('ATL08 REL006 Classifications')
    
    axes[1].plot(alongtrack[atl24_class == 0],h_ph[atl24_class == 0],'.',color=[0.8,0.8,0.8],label='Unclassified')
    axes[1].plot(alongtrack[atl24_class == 41],h_ph[atl24_class == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='Water Surface (41)')
    axes[1].plot(alongtrack[atl24_class == 40],h_ph[atl24_class == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='Bathymetry (40)')
    # axes[1].set_xlabel('Alongtrack (m)')
    axes[1].set_ylabel('Orthometric Height (m)')
    axes[1].legend(markerscale=3)
    axes[1].set_title('ATL24 REL001 Classifications')
    
    axes[2].plot(alongtrack[contested_class == 0],h_ph[contested_class == 0],'.',color=[0.8,0.8,0.8],label='Unclassified Photon')
    axes[2].plot(alongtrack[contested_class == 1],h_ph[contested_class == 1],'.',color=[0.12156863, 0.81960784, 0.12156863],label='ATL08 Photon')
    axes[2].plot(alongtrack[contested_class == 2],h_ph[contested_class == 2],'.',color=[0.        , 0.61568627, 0.96862745],label='ATL24 Photon')
    axes[2].plot(alongtrack[contested_class == 3],h_ph[contested_class == 3],'.',color=[1,0,0],label='Contested Photon')
    axes[2].set_xlabel('Alongtrack (m)')
    axes[2].set_ylabel('Orthometric Height (m)')
    axes[2].legend(markerscale=3)
    axes[2].set_title('Contested Photons')
    
    
    fig.suptitle(title, fontsize=16)
    plt.show()
    
    
def plot_alongtrack_als(df_als,title):
    plt.figure()
    plt.plot(df_als.alongtrack[df_als.classification == 7],df_als.ortho_h[df_als.classification == 7],'.',color=[0.8,0,0],label='Noise (7)')
    plt.plot(df_als.alongtrack[df_als.classification == 1],df_als.ortho_h[df_als.classification == 1],'.',color=[0.8, 0.8, 0.8],label='Unclassified (1)')
    plt.plot(df_als.alongtrack[df_als.classification == 2],df_als.ortho_h[df_als.classification == 2],'.',color=[0.69803922, 0.44313725, 0.23921569],label='Terrain (2)')
    plt.plot(df_als.alongtrack[df_als.classification == 45],df_als.ortho_h[df_als.classification == 45],'.',color=[0.65098039, 0.98039215, 1.0],label='Water column (45)')
    plt.plot(df_als.alongtrack[df_als.classification == 40],df_als.ortho_h[df_als.classification == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='Bathymetry (40)')
    plt.plot(df_als.alongtrack[df_als.classification == 41],df_als.ortho_h[df_als.classification == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='Water Surface (41)')
    plt.xlabel('Alongtrack (m)')
    plt.ylabel('Height (m)')
    plt.legend(markerscale=3)
    plt.title(title)
    plt.show()
    
    
    
def plot_alongtrack_atl08_als(alongtrack,h_ph,atl08_class,title):
    plt.figure()
    plt.plot(alongtrack[atl08_class == 0],h_ph[atl08_class == 0],'.',color=[0.8,0.8,0.8],label='Unclassified')
    plt.plot(alongtrack[atl08_class == 3],h_ph[atl08_class == 3],'.',color=[0.20392157, 0.70196078, 0.20392157],label='Top of Canopy (3)')
    plt.plot(alongtrack[atl08_class == 2],h_ph[atl08_class == 2],'.',color=[0.12156863, 0.41960784, 0.12156863],label='Canopy (2)')
    plt.plot(alongtrack[atl08_class == 1],h_ph[atl08_class == 1],'.',color=[0.69803922, 0.44313725, 0.23921569],label='Topography (1)')
    plt.xlabel('Alongtrack (m)')
    plt.ylabel('Orthometric Height (m)')
    plt.legend(markerscale=3)
    plt.title(title)
    plt.show()
    
    
def plot_alongtrack_atl_als(df_ph, df_als,title):
    plt.figure()
    # plt.plot(df_als.alongtrack[df_als.classification == 7],df_als.ortho_h[df_als.classification == 7],'.',color=[0.8,0,0],label='Noise (7)')
    # plt.plot(df_als.alongtrack[df_als.classification.isin([41,45])],df_als.ortho_h[df_als.classification.isin([41,45])],'.',color=[0.4, 0.4, 0.8],label='ALS Water')
    plt.plot(df_als.alongtrack[df_als.classification == 1],df_als.ortho_h[df_als.classification == 1],'.',color=[0.8, 0.8, 0.8],label='ALS Unclassified (Veg)')
    plt.plot(df_als.alongtrack[df_als.classification.isin([2,40])],df_als.ortho_h[df_als.classification.isin([2,40])],'.',color=[0.4, 0.4, 0.4],label='ALS Topobathy')
    
    plt.plot(df_ph.alongtrack[df_ph.combined_class == 0],df_ph.ortho_h[df_ph.combined_class == 0],'.',color=[0.8,0.8,1],label='Unclassified')
    # plt.plot(df_ph.alongtrack[df_ph.combined_class == 3],df_ph.ortho_h[df_ph.combined_class == 3],'.',color=[0.20392157, 0.70196078, 0.20392157],label='Top of Canopy (3)')
    plt.plot(df_ph.alongtrack[df_ph.combined_class.isin([2,3])],df_ph.ortho_h[df_ph.combined_class.isin([2,3])],'.',color=[0.12156863, 0.41960784, 0.12156863],label='ATL08 Canopy')
    plt.plot(df_ph.alongtrack[df_ph.combined_class == 1],df_ph.ortho_h[df_ph.combined_class == 1],'.',color=[0.69803922, 0.44313725, 0.23921569],label='ATL08 Terrain')
    plt.plot(df_ph.alongtrack[df_ph.combined_class == 41],df_ph.ortho_h[df_ph.combined_class == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='ATL24 Water Surface')
    plt.plot(df_ph.alongtrack[df_ph.combined_class == 40],df_ph.ortho_h[df_ph.combined_class == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='ATL24 Bathymetry')
    plt.xlabel('Alongtrack (m)')
    plt.ylabel('Orthometric Height (m)')
    plt.legend(markerscale=3)
    plt.title(title)
    plt.show()

def plot_alongtrack_atl_als2(df_ph, df_als,title):
    plt.figure()
    # plt.plot(df_als.alongtrack[df_als.classification == 7],df_als.ortho_h[df_als.classification == 7],'.',color=[0.8,0,0],label='Noise (7)')
    plt.plot(df_als.alongtrack[df_als.classification.isin([41,45])],df_als.ellip_h[df_als.classification.isin([41,45])],'.',color=[0.4, 0.4, 0.6],label='ALS Water')
    plt.plot(df_als.alongtrack[df_als.classification == 1],df_als.ellip_h[df_als.classification == 1],'.',color=[0.7, 0.9, 0.7],label='ALS Unclassified (Veg)')
    plt.plot(df_als.alongtrack[df_als.classification.isin([2,40])],df_als.ellip_h[df_als.classification.isin([2,40])],'.',color=[0.4, 0.4, 0.4],label='ALS Topobathy')
    
    plt.plot(df_ph.alongtrack[df_ph.combined_class == 0],df_ph.ortho_h[df_ph.combined_class == 0],'.',color=[0.8,0.8,0.8],label='Unclassified')
    # plt.plot(df_ph.alongtrack[df_ph.combined_class == 3],df_ph.ortho_h[df_ph.combined_class == 3],'.',color=[0.20392157, 0.70196078, 0.20392157],label='Top of Canopy (3)')
    plt.plot(df_ph.alongtrack[df_ph.combined_class.isin([2,3])],df_ph.ortho_h[df_ph.combined_class.isin([2,3])],'.',color=[0.12156863, 0.41960784, 0.12156863],label='ATL08 Canopy')
    plt.plot(df_ph.alongtrack[df_ph.combined_class == 1],df_ph.ortho_h[df_ph.combined_class == 1],'.',color=[0.69803922, 0.44313725, 0.23921569],label='ATL08 Terrain')
    plt.plot(df_ph.alongtrack[df_ph.combined_class == 41],df_ph.ortho_h[df_ph.combined_class == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='ATL24 Water Surface')
    plt.plot(df_ph.alongtrack[df_ph.combined_class == 40],df_ph.ortho_h[df_ph.combined_class == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='ATL24 Bathymetry')
    plt.xlabel('Alongtrack (m)')
    plt.ylabel('Orthometric Height (m)')
    plt.legend(markerscale=3)
    plt.title(title)
    plt.show()
    
def plot_alongtrack_atl_als_dual(df_ph, df_als,title):
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    # axes[0].plot(df_als.alongtrack[df_als.classification == 7],df_als.ellip_h[df_als.classification == 7],'.',color=[0.8,0,0],label='Noise (7)')
    axes[0].plot(df_als.alongtrack[df_als.classification.isin([41,45])],df_als.ellip_h[df_als.classification.isin([41,45])],'.',color=[0.4, 0.4, 0.6],label='ALS Water')
    axes[0].plot(df_als.alongtrack[df_als.classification == 1],df_als.ellip_h[df_als.classification == 1],'.',color=[0.7, 0.9, 0.7],label='ALS Unclassified (Veg)')
    axes[0].plot(df_als.alongtrack[df_als.classification.isin([2,40])],df_als.ellip_h[df_als.classification.isin([2,40])],'.',color=[0.4, 0.4, 0.4],label='ALS Topobathy')
    axes[0].set_ylabel('Orthometric Height (m)')
    axes[0].legend(markerscale=3)
    axes[0].set_title('ALS Profile')
    
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 0],df_ph.ortho_h[df_ph.combined_class == 0],'.',color=[0.8,0.8,0.9],label='Unclassified')
    # axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 3],df_ph.ortho_h[df_ph.combined_class == 3],'.',color=[0.20392157, 0.70196078, 0.20392157],label='Top of Canopy (3)')
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class.isin([2,3])],df_ph.ortho_h[df_ph.combined_class.isin([2,3])],'.',color=[0.12156863, 0.41960784, 0.12156863],label='ATL08 Canopy')
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 1],df_ph.ortho_h[df_ph.combined_class == 1],'.',color=[0.69803922, 0.44313725, 0.23921569],label='ATL08 Terrain')
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 41],df_ph.ortho_h[df_ph.combined_class == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='ATL24 Water Surface')
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 40],df_ph.ortho_h[df_ph.combined_class == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='ATL24 Bathymetry')
    axes[1].set_xlabel('Alongtrack (m)')
    axes[1].set_ylabel('Orthometric Height (m)')
    axes[1].legend(markerscale=3)
    axes[1].set_title('ICESat-2 Profile')
    
    fig.suptitle(title, fontsize=16)
    
    plt.show()
    
    
def plot_alongtrack_atl08_als_dual(df_ph1, df_ph2, df_als,title):
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    # axes[0].plot(df_als.alongtrack[df_als.classification == 7],df_als.ellip_h[df_als.classification == 7],'.',color=[0.8,0,0],label='Noise (7)')
    # axes[0].plot(df_als.alongtrack[df_als.classification.isin([41,45])],df_als.ellip_h[df_als.classification.isin([41,45])],'.',color=[0.4, 0.4, 0.6],label='ALS Water')
    axes[0].plot(df_als.alongtrack[df_als.classification == 1],df_als.ellip_h[df_als.classification == 1],'.',color=[0.9, 0.9, 0.9],label='ALS Unclassified')
    axes[0].plot(df_als.alongtrack[df_als.classification.isin([3,4,5])],df_als.ellip_h[df_als.classification.isin([3,4,5])],'.',color=[0.7, 0.7, 0.7],label='ALS Canopy')
    axes[0].plot(df_als.alongtrack[df_als.classification.isin([2,40])],df_als.ellip_h[df_als.classification.isin([2,40])],'.',color=[0.4, 0.4, 0.4],label='ALS Terrain')
    axes[0].plot(df_ph1.alongtrack[df_ph1.atl08_class == 0],df_ph1.h_ph[df_ph1.atl08_class == 0],'.',color=[0.0,0.9,0.9],alpha = 0.2,label='ATL08 Unclassified')
    axes[0].plot(df_ph1.alongtrack[df_ph1.atl08_class == 3],df_ph1.h_ph[df_ph1.atl08_class == 3],'.',color=[0.20392157, 0.70196078, 0.20392157],label='ATL08 Top of Canopy')
    axes[0].plot(df_ph1.alongtrack[df_ph1.atl08_class.isin([2])],df_ph1.h_ph[df_ph1.atl08_class.isin([2])],'.',color=[0.12156863, 0.41960784, 0.12156863],label='ATL08 Canopy')
    axes[0].plot(df_ph1.alongtrack[df_ph1.atl08_class == 1],df_ph1.h_ph[df_ph1.atl08_class == 1],'.',color=[0.69803922, 0.44313725, 0.23921569],label='ATL08 Terrain')
    axes[0].set_ylabel('Height (m)')
    axes[0].legend(markerscale=3)
    axes[0].set_title('ATL08 REL006')
        
    axes[1].plot(df_als.alongtrack[df_als.classification == 1],df_als.ellip_h[df_als.classification == 1],'.',color=[0.9, 0.9, 0.9],label='ALS Unclassified')
    axes[1].plot(df_als.alongtrack[df_als.classification.isin([3,4,5])],df_als.ellip_h[df_als.classification.isin([3,4,5])],'.',color=[0.7, 0.7, 0.7],label='ALS Canopy')
    axes[1].plot(df_als.alongtrack[df_als.classification.isin([2,40])],df_als.ellip_h[df_als.classification.isin([2,40])],'.',color=[0.4, 0.4, 0.4],label='ALS Terrain')
    axes[1].plot(df_ph2.alongtrack[df_ph2.atl08_class == 0],df_ph2.h_ph[df_ph2.atl08_class == 0],'.',color=[0.0,0.9,0.9],alpha = 0.2, label='ATL08 Unclassified')
    axes[1].plot(df_ph2.alongtrack[df_ph2.atl08_class == 3],df_ph2.h_ph[df_ph2.atl08_class == 3],'.',color=[0.20392157, 0.70196078, 0.20392157],label='ATL08 Top of Canopy')
    axes[1].plot(df_ph2.alongtrack[df_ph2.atl08_class.isin([2])],df_ph2.h_ph[df_ph2.atl08_class.isin([2])],'.',color=[0.12156863, 0.41960784, 0.12156863],label='ATL08 Canopy')
    axes[1].plot(df_ph2.alongtrack[df_ph2.atl08_class == 1],df_ph2.h_ph[df_ph2.atl08_class == 1],'.',color=[0.69803922, 0.44313725, 0.23921569],label='ATL08 Terrain')
    
    axes[1].set_xlabel('Alongtrack (m)')
    axes[1].set_ylabel('Height (m)')
    # axes[1].legend(markerscale=3)
    axes[1].set_title('ATL08 REL007')
    
    fig.suptitle(title, fontsize=16)
    
    plt.show()        
    
def plot_alongtrack_atl_als_tri(df_ph, df_als,title):
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    # axes[0].plot(df_als.alongtrack[df_als.classification == 7],df_als.ellip_h[df_als.classification == 7],'.',color=[0.8,0,0],label='Noise (7)')
    axes[0].plot(df_als.alongtrack[df_als.classification.isin([41,45])],df_als.ellip_h[df_als.classification.isin([41,45])],'.',color=[0.4, 0.4, 0.6],label='ALS Water')
    axes[0].plot(df_als.alongtrack[df_als.classification == 1],df_als.ellip_h[df_als.classification == 1],'.',color=[0.7, 0.9, 0.7],label='ALS Unclassified (Veg)')
    axes[0].plot(df_als.alongtrack[df_als.classification.isin([2,40])],df_als.ellip_h[df_als.classification.isin([2,40])],'.',color=[0.4, 0.4, 0.4],label='ALS Topobathy')
    axes[0].set_ylabel('Orthometric Height (m)')
    axes[0].legend(markerscale=3)
    axes[0].set_title('ALS Profile')
    
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 0],df_ph.ortho_h[df_ph.combined_class == 0],'.',color=[0.8,0.8,0.9],label='Unclassified')
    # axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 3],df_ph.ortho_h[df_ph.combined_class == 3],'.',color=[0.20392157, 0.70196078, 0.20392157],label='Top of Canopy (3)')
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class.isin([2,3])],df_ph.ortho_h[df_ph.combined_class.isin([2,3])],'.',color=[0.12156863, 0.41960784, 0.12156863],label='Canopy (2)')
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 1],df_ph.ortho_h[df_ph.combined_class == 1],'.',color=[0.69803922, 0.44313725, 0.23921569],label='Topography (1)')
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 41],df_ph.ortho_h[df_ph.combined_class == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='Water Surface (41)')
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 40],df_ph.ortho_h[df_ph.combined_class == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='Bathymetry (40)')
    # axes[1].set_xlabel('Alongtrack (m)')
    axes[1].set_ylabel('Orthometric Height (m)')
    axes[1].legend(markerscale=3)
    axes[1].set_title('ICESat-2 Profile')
    
    
    axes[2].plot(df_als.alongtrack[df_als.classification.isin([41,45])],df_als.ellip_h[df_als.classification.isin([41,45])],'.',color=[0.4, 0.4, 0.6],label='ALS Water')
    axes[2].plot(df_als.alongtrack[df_als.classification == 1],df_als.ellip_h[df_als.classification == 1],'.',color=[0.7, 0.9, 0.7],label='ALS Unclassified (Veg)')
    axes[2].plot(df_als.alongtrack[df_als.classification.isin([2,40])],df_als.ellip_h[df_als.classification.isin([2,40])],'.',color=[0.4, 0.4, 0.4],label='ALS Topobathy')
    
    axes[2].plot(df_ph.alongtrack[df_ph.combined_class == 0],df_ph.ortho_h[df_ph.combined_class == 0],'.',color=[0.8,0.8,0.9],label='Unclassified')
    # axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 3],df_ph.ortho_h[df_ph.combined_class == 3],'.',color=[0.20392157, 0.70196078, 0.20392157],label='Top of Canopy (3)')
    axes[2].plot(df_ph.alongtrack[df_ph.combined_class.isin([2,3])],df_ph.ortho_h[df_ph.combined_class.isin([2,3])],'.',color=[0.12156863, 0.41960784, 0.12156863],label='Canopy (2)')
    axes[2].plot(df_ph.alongtrack[df_ph.combined_class == 1],df_ph.ortho_h[df_ph.combined_class == 1],'.',color=[0.69803922, 0.44313725, 0.23921569],label='Topography (1)')
    axes[2].plot(df_ph.alongtrack[df_ph.combined_class == 41],df_ph.ortho_h[df_ph.combined_class == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='Water Surface (41)')
    axes[2].plot(df_ph.alongtrack[df_ph.combined_class == 40],df_ph.ortho_h[df_ph.combined_class == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='Bathymetry (40)')
    axes[2].set_xlabel('Alongtrack (m)')
    axes[2].set_ylabel('Orthometric Height (m)')
    axes[2].legend(markerscale=3)
    axes[2].set_title('ALS + ICESat-2 Profile')
    
    fig.suptitle(title, fontsize=16)
    
    plt.show()
    
def plot_y_als(df_als,title):
    plt.figure()
    plt.plot(df_als.y[df_als.classification == 7],df_als.ortho_h[df_als.classification == 7],'.',color=[0.8,0,0],label='Noise (7)')
    plt.plot(df_als.y[df_als.classification == 1],df_als.ortho_h[df_als.classification == 1],'.',color=[0.8, 0.8, 0.8],label='Unclassified (1)')
    plt.plot(df_als.y[df_als.classification == 2],df_als.ortho_h[df_als.classification == 2],'.',color=[0.69803922, 0.44313725, 0.23921569],label='Ground (2)')
    plt.plot(df_als.y[df_als.classification == 45],df_als.ortho_h[df_als.classification == 45],'.',color=[0.65098039, 0.98039215, 1.0],label='Water column (45)')
    plt.plot(df_als.y[df_als.classification == 40],df_als.ortho_h[df_als.classification == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='Bathymetry (40)')
    plt.plot(df_als.y[df_als.classification == 41],df_als.ortho_h[df_als.classification == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='Water Surface (41)')
    plt.xlabel('Alongtrack (m)')
    plt.ylabel('Orthometric Height (m)')
    plt.legend(markerscale=3)
    plt.title(title)
    plt.show()
    
def plot_with_secondary_axis(df_als):
    """
    Creates a static plot with a secondary x-axis at the top to show
    corresponding longitude values at specific latitude ticks.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the main data (Elevation vs. Latitude)
    # ax1.plot(df['latitude'], df['elevation'], '.-', markersize=4, label='Elevation Profile')

    ax1.plot(df_als.latitude[df_als.classification == 7],df_als.ortho_h[df_als.classification == 7],'.',color=[0.8,0,0],label='Noise (7)')
    ax1.plot(df_als.latitude[df_als.classification == 1],df_als.ortho_h[df_als.classification == 1],'.',color=[0.8, 0.8, 0.8],label='Unclassified (1)')
    ax1.plot(df_als.latitude[df_als.classification == 2],df_als.ortho_h[df_als.classification == 2],'.',color=[0.69803922, 0.44313725, 0.23921569],label='Ground (2)')
    ax1.plot(df_als.latitude[df_als.classification == 45],df_als.ortho_h[df_als.classification == 45],'.',color=[0.65098039, 0.98039215, 1.0],label='Water column (45)')
    ax1.plot(df_als.latitude[df_als.classification == 40],df_als.ortho_h[df_als.classification == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='Bathymetry (40)')
    plt.plot(df_als.latitude[df_als.classification == 41],df_als.ortho_h[df_als.classification == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='Water Surface (41)')    


    ax1.set_xlabel('Latitude (Decimal Degrees)')
    ax1.set_ylabel('Elevation above Ellipsoid (m)')
    ax1.set_title('Elevation Profile with Longitude Axis')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Create the secondary (parasite) x-axis for Longitude ---
    ax2 = ax1.twiny() # Create a new x-axis that shares the same y-axis

    # Set the limits of the new axis to match the first one
    ax2.set_xlim(ax1.get_xlim())
    
    # Get the tick locations from the primary latitude axis
    latitude_ticks = ax1.get_xticks()
    
    # Find the corresponding longitude values for each latitude tick
    # We use np.interp for a linear interpolation
    longitude_labels = np.interp(latitude_ticks, df_als['latitude'], df_als['longitude'])
    
    # Set the ticks and labels for the secondary axis
    ax2.set_xticks(latitude_ticks)
    ax2.set_xticklabels([f'{lon:.3f}' for lon in longitude_labels])
    ax2.set_xlabel('Corresponding Longitude (Decimal Degrees)')
    
    fig.tight_layout()
    plt.show()
    
def plot_with_secondary_axis2(df_als):
    """
    Creates a static plot with a secondary x-axis at the top to show
    corresponding longitude values at specific latitude ticks.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the main data (Elevation vs. Latitude)
    # ax1.plot(df['latitude'], df['elevation'], '.-', markersize=4, label='Elevation Profile')

    ax1.plot(df_als.longitude[df_als.classification == 7],df_als.ortho_h[df_als.classification == 7],'.',color=[0.8,0,0],label='Noise (7)')
    ax1.plot(df_als.longitude[df_als.classification == 1],df_als.ortho_h[df_als.classification == 1],'.',color=[0.8, 0.8, 0.8],label='Unclassified (1)')
    ax1.plot(df_als.longitude[df_als.classification == 2],df_als.ortho_h[df_als.classification == 2],'.',color=[0.69803922, 0.44313725, 0.23921569],label='Ground (2)')
    ax1.plot(df_als.longitude[df_als.classification == 45],df_als.ortho_h[df_als.classification == 45],'.',color=[0.65098039, 0.98039215, 1.0],label='Water column (45)')
    ax1.plot(df_als.longitude[df_als.classification == 40],df_als.ortho_h[df_als.classification == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='Bathymetry (40)')
    plt.plot(df_als.longitude[df_als.classification == 41],df_als.ortho_h[df_als.classification == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='Water Surface (41)')    


    ax1.set_xlabel('Longitude (Decimal Degrees)')
    ax1.set_ylabel('Elevation above Ellipsoid (m)')
    ax1.set_title('Elevation Profile with Longitude Axis')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Create the secondary (parasite) x-axis for Longitude ---
    ax2 = ax1.twiny() # Create a new x-axis that shares the same y-axis

    # Set the limits of the new axis to match the first one
    ax2.set_xlim(ax1.get_xlim())
    
    # Get the tick locations from the primary latitude axis
    longitude_ticks = ax1.get_xticks()
    
    # Find the corresponding longitude values for each latitude tick
    # We use np.interp for a linear interpolation
    latitude_labels = np.interp(longitude_ticks, df_als['longitude'], df_als['latitude'])
    
    # Set the ticks and labels for the secondary axis
    ax2.set_xticks(longitude_ticks)
    ax2.set_xticklabels([f'{lat:.3f}' for lat in latitude_labels])
    ax2.set_xlabel('Corresponding Longitude (Decimal Degrees)')
    
    fig.tight_layout()
    plt.show()
    
    
    
def plot_alongtrack_atl_als_alongtrack_versions(df_ph, df_als,title):
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=True)

    
    axes[0].plot(df_als.alongtrack[df_als.classification.isin([41,45])],df_als.h_norm[df_als.classification.isin([41,45])],'.',color=[0.4, 0.4, 0.6],label='ALS Water')
    axes[0].plot(df_als.alongtrack[df_als.classification == 1],df_als.h_norm[df_als.classification == 1],'.',color=[0.7, 0.9, 0.7],label='ALS Unclassified (Veg)')
    axes[0].plot(df_als.alongtrack[df_als.classification.isin([2,40])],df_als.h_norm[df_als.classification.isin([2,40])],'.',color=[0.4, 0.4, 0.4],label='ALS Topobathy')
    
    axes[0].plot(df_ph.alongtrack[df_ph.combined_class == 0],df_ph.h_norm[df_ph.combined_class == 0],'.',color=[0.8,0.8,0.9],label='Unclassified (0)')
    # axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 3],df_ph.ortho_h[df_ph.combined_class == 3],'.',color=[0.20392157, 0.70196078, 0.20392157],label='Top of Canopy (3)')
    axes[0].plot(df_ph.alongtrack[df_ph.combined_class.isin([2,3])],df_ph.h_norm[df_ph.combined_class.isin([2,3])],'.',color=[0.12156863, 0.41960784, 0.12156863],label='Canopy (2)')
    axes[0].plot(df_ph.alongtrack[df_ph.combined_class == 1],df_ph.h_norm[df_ph.combined_class == 1],'.',color=[0.69803922, 0.44313725, 0.23921569],label='Topography (1)')
    axes[0].plot(df_ph.alongtrack[df_ph.combined_class == 41],df_ph.h_norm[df_ph.combined_class == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='Water Surface (41)')
    axes[0].plot(df_ph.alongtrack[df_ph.combined_class == 40],df_ph.h_norm[df_ph.combined_class == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='Bathymetry (40)')
    axes[0].set_xlabel('Alongtrack (m)')
    axes[0].set_ylabel('Orthometric Height (m)')
    axes[0].legend(markerscale=3)
    axes[0].set_title('h_norm')
    
    axes[1].plot(df_als.alongtrack[df_als.classification.isin([41,45])],df_als.h_norm[df_als.classification.isin([41,45])],'.',color=[0.4, 0.4, 0.6],label='ALS Water')
    axes[1].plot(df_als.alongtrack[df_als.classification == 1],df_als.h_norm[df_als.classification == 1],'.',color=[0.7, 0.9, 0.7],label='ALS Unclassified (Veg)')
    axes[1].plot(df_als.alongtrack[df_als.classification.isin([2,40])],df_als.h_norm[df_als.classification.isin([2,40])],'.',color=[0.4, 0.4, 0.4],label='ALS Topobathy')
    
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 0],df_ph.h_te_norm[df_ph.combined_class == 0],'.',color=[0.8,0.8,0.9],label='Unclassified (0)')
    # axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 3],df_ph.ortho_h[df_ph.combined_class == 3],'.',color=[0.20392157, 0.70196078, 0.20392157],label='Top of Canopy (3)')
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class.isin([2,3])],df_ph.h_te_norm[df_ph.combined_class.isin([2,3])],'.',color=[0.12156863, 0.41960784, 0.12156863],label='Canopy (2)')
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 1],df_ph.h_te_norm[df_ph.combined_class == 1],'.',color=[0.69803922, 0.44313725, 0.23921569],label='Topography (1)')
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 41],df_ph.h_te_norm[df_ph.combined_class == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='Water Surface (41)')
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 40],df_ph.h_te_norm[df_ph.combined_class == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='Bathymetry (40)')
    axes[1].set_xlabel('Alongtrack (m)')
    axes[1].set_ylabel('Orthometric Height (m)')
    axes[1].legend(markerscale=3)
    axes[1].set_title('h_te_norm')
    
    axes[2].plot(df_als.alongtrack[df_als.classification.isin([41,45])],df_als.h_norm[df_als.classification.isin([41,45])],'.',color=[0.4, 0.4, 0.6],label='ALS Water')
    axes[2].plot(df_als.alongtrack[df_als.classification == 1],df_als.h_norm[df_als.classification == 1],'.',color=[0.7, 0.9, 0.7],label='ALS Unclassified (Veg)')
    axes[2].plot(df_als.alongtrack[df_als.classification.isin([2,40])],df_als.h_norm[df_als.classification.isin([2,40])],'.',color=[0.4, 0.4, 0.4],label='ALS Topobathy')
    
    axes[2].plot(df_ph.alongtrack[df_ph.combined_class == 0],df_ph.h_toposurf_norm[df_ph.combined_class == 0],'.',color=[0.8,0.8,0.9],label='Unclassified')
    # axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 3],df_ph.ortho_h[df_ph.combined_class == 3],'.',color=[0.20392157, 0.70196078, 0.20392157],label='Top of Canopy (3)')
    axes[2].plot(df_ph.alongtrack[df_ph.combined_class.isin([2,3])],df_ph.h_toposurf_norm[df_ph.combined_class.isin([2,3])],'.',color=[0.12156863, 0.41960784, 0.12156863],label='Canopy (2)')
    axes[2].plot(df_ph.alongtrack[df_ph.combined_class == 1],df_ph.h_toposurf_norm[df_ph.combined_class == 1],'.',color=[0.69803922, 0.44313725, 0.23921569],label='Topography (1)')
    axes[2].plot(df_ph.alongtrack[df_ph.combined_class == 41],df_ph.h_toposurf_norm[df_ph.combined_class == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='Water Surface (41)')
    axes[2].plot(df_ph.alongtrack[df_ph.combined_class == 40],df_ph.h_toposurf_norm[df_ph.combined_class == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='Bathymetry (40)')
    axes[2].set_xlabel('Alongtrack (m)')
    axes[2].set_ylabel('Orthometric Height (m)')
    axes[2].legend(markerscale=3)
    axes[2].set_title('h_toposurf_norm')
    
    fig.suptitle(title, fontsize=16)
    
    plt.show()
    
    
    
def plot_alongtrack_atl_als_alongtrack_versions2(df_ph, df_als,title):
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)

    
    axes[0].plot(df_als.alongtrack[df_als.classification.isin([41,45])],df_als.h_norm[df_als.classification.isin([41,45])],'.',color=[0.4, 0.4, 0.6],label='ALS Water')
    axes[0].plot(df_als.alongtrack[df_als.classification == 1],df_als.h_norm[df_als.classification == 1],'.',color=[0.7, 0.9, 0.7],label='ALS Unclassified (Veg)')
    axes[0].plot(df_als.alongtrack[df_als.classification.isin([2,40])],df_als.h_norm[df_als.classification.isin([2,40])],'.',color=[0.4, 0.4, 0.4],label='ALS Topobathy')
    
    axes[0].plot(df_ph.alongtrack[df_ph.combined_class == 0],df_ph.h_norm[df_ph.combined_class == 0],'.',color=[0.8,0.8,0.9],label='Unclassified (0)')
    # axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 3],df_ph.ortho_h[df_ph.combined_class == 3],'.',color=[0.20392157, 0.70196078, 0.20392157],label='Top of Canopy (3)')
    axes[0].plot(df_ph.alongtrack[df_ph.combined_class.isin([2,3])],df_ph.h_norm[df_ph.combined_class.isin([2,3])],'.',color=[0.12156863, 0.41960784, 0.12156863],label='Canopy (2)')
    axes[0].plot(df_ph.alongtrack[df_ph.combined_class == 1],df_ph.h_norm[df_ph.combined_class == 1],'.',color=[0.69803922, 0.44313725, 0.23921569],label='Topography (1)')
    axes[0].plot(df_ph.alongtrack[df_ph.combined_class == 41],df_ph.h_norm[df_ph.combined_class == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='Water Surface (41)')
    axes[0].plot(df_ph.alongtrack[df_ph.combined_class == 40],df_ph.h_norm[df_ph.combined_class == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='Bathymetry (40)')
    axes[0].set_xlabel('Alongtrack (m)')
    axes[0].set_ylabel('Relative Height (m)')
    axes[0].legend(markerscale=3)
    axes[0].set_title('Terrain Normalized')
    
    axes[1].plot(df_als.alongtrack[df_als.classification.isin([41,45])],df_als.h_norm[df_als.classification.isin([41,45])],'.',color=[0.4, 0.4, 0.6],label='ALS Water')
    axes[1].plot(df_als.alongtrack[df_als.classification == 1],df_als.h_norm[df_als.classification == 1],'.',color=[0.7, 0.9, 0.7],label='ALS Unclassified (Veg)')
    axes[1].plot(df_als.alongtrack[df_als.classification.isin([2,40])],df_als.h_norm[df_als.classification.isin([2,40])],'.',color=[0.4, 0.4, 0.4],label='ALS Topobathy')
    
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 0],df_ph.h_topobathy_norm[df_ph.combined_class == 0],'.',color=[0.8,0.8,0.9],label='Unclassified (0)')
    # axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 3],df_ph.ortho_h[df_ph.combined_class == 3],'.',color=[0.20392157, 0.70196078, 0.20392157],label='Top of Canopy (3)')
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class.isin([2,3])],df_ph.h_topobathy_norm[df_ph.combined_class.isin([2,3])],'.',color=[0.12156863, 0.41960784, 0.12156863],label='Canopy (2)')
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 1],df_ph.h_topobathy_norm[df_ph.combined_class == 1],'.',color=[0.69803922, 0.44313725, 0.23921569],label='Topography (1)')
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 41],df_ph.h_topobathy_norm[df_ph.combined_class == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='Water Surface (41)')
    axes[1].plot(df_ph.alongtrack[df_ph.combined_class == 40],df_ph.h_topobathy_norm[df_ph.combined_class == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='Bathymetry (40)')
    axes[1].set_xlabel('Alongtrack (m)')
    axes[1].set_ylabel('Relative Height (m)')
    axes[1].legend(markerscale=3)
    axes[1].set_title('Topobathy Normalized')
    
    
    fig.suptitle(title, fontsize=16)
    
    plt.show()
