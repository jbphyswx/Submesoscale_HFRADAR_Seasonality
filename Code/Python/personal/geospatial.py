# -*- coding: utf-8 -*-

""" 01/22/2020, Jordan Benjamin """



import numpy as np
import metpy.constants as mpconc
import pint
ureg = pint.UnitRegistry()

import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import shapely.geometry
from   shapely.geometry import Point
from   shapely.prepared import prep


import personal.data_structures
import personal.constants

r =  personal.constants.earth['radius']    # earth radius meters,  # r = mpconc.earth_avg_radius

#avor = mpcalc.vorticity(uwnd_500, vwnd_500, dx, dy, dim_order='yx') + f

def lat_lon_to_x_y_deltas(lon,lat):
    """ lat/lon could be vectors or meshgrids  (both one or the other, or code might fail)
        If are matricess, assumes lat varies in dimension 1, lon in dimension 2 (i.e. if printed on screen would match Earth's orientation)
        Assumes uniform spacing in lat/lon coords, lat/lon in degrees
        Converts to cartesian, returns meshgrids (must be to define x_0,y_0) w/ x_0 y_0 at bottom left of input arrays
        use mpcalc lat_lon_grid deltas as equivalent
        """
        
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    if personal.data_structures.isvector(lat):
        lat,lon = np.meshgrid(lat,lon,indexing='ij') # works for lists or arrays, matrix indexing so rows are lat, columns are lon
    
    dlat = np.diff(lat,axis=0) # difference between latitudes
    dy   = r * dlat            # r imbues meter units
    #
    dlon = np.diff(lon,axis=1) # difference between longitudes
    dx = r*np.cos(lat[:,:-1]) * dlon # take the lats only so far in lon 
        
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


def is_land(lon,lat, land = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m') ):
    """
    Calculates if coordinates are land (adapted from https://stackoverflow.com/questions/53322952/creating-a-land-ocean-mask-in-cartopy)
    """
    land_polygons = [prep(land_polygon) for land_polygon in  list(land.geometries())] # preparing really speeds things up per link above
    
    coords = np.stack((lon,lat),axis=0)
    points = np.apply_along_axis(lambda x: np.array([Point(x)], dtype=object),0, coords ).squeeze(axis=0) # use lambda fcn to force apply_along_axis to return objects inside arrays, then squeeze that new axis


    is_land = np.full(coords.shape[1:],False) # start off assuming all are land
    for land_polygon in land_polygons:
        is_land = np.logical_or(is_land, np.vectorize(land_polygon.covers)(points)) # points that aren't will fall off
#         is_land = np.logical_or(is_land, np.array([land_polygon.covers(x) for x in points])) # points that aren't will fall off

    return is_land
    


