# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:17:52 2020

@author: Jordan
"""

import numpy as np
import metpy.calc as mpcalc

import pint
ureg = pint.UnitRegistry()

import personal.data_structures


def lat_lon_to_x_y_deltas(lon,lat): # Exists in metpy database
    
    if personal.data_structures.isvector(lat):
        lat,lon = np.meshgrid(lat,lon,indexing='ij') # works for lists or arrays, matrix indexing so rows are lat, columns are lon
        
    dx,dy = mpcalc.lat_lon_grid_deltas(lon,lat) # defaults to x axis is -1 (last axis, i.e. columns), y-axis is -2 (first axis, i.e. rows)
    return dx,dy


def lat_lon_to_x_y(lon,lat):
    """ lat/lon could be vectors or meshgrids  (both one or the other, or code might fail)
        If are matricess, assumes lat varies in dimension 1, lon in dimension 2 (i.e. if printed on screen would match Earth's orientation)
        Assumes uniform spacing in lat/lon coords, lat/lon in degrees
        Converts to cartesian, returns meshgrids (must be to define x_0,y_0) w/ x_0 y_0 at bottom left of input arrays"""

    dx,dy = lat_lon_to_x_y_deltas(lon,lat)
    sz = [np.shape(dx)[0],np.shape(dy)[1]] # could be either, dx ~ dlon, dy~dlat, size is [#lat, #lon], select size that wasn't differenced
    
    x = np.hstack( (np.zeros((sz[0], 1)) , np.cumsum(dx, axis = 1)) ) * ureg.meter # doing out here bc stack seems to lose units
    y = np.vstack( (np.zeros((1, sz[1])) , np.cumsum(dy, axis = 0)) ) * ureg.meter
    
    return x,y