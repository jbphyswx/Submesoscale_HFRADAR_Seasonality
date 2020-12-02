# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:39:57 2020

@author: Jordan

Note:

"One unfortunate limitation of using datetime64[ns] is that it limits the native representation of dates to those that fall between the years 1678 and 2262. When a netCDF file contains dates outside of these bounds, dates will be returned as arrays of netcdftime.datetime objects."
"""

import pandas as pd
import numpy as np
import xarray as xr
import dateutil
import datetime



import personal.data_structures

def get_np_datetime64_attrs(date_arr,attrs,dict_output=False):
    """
    attrs is a list, dict etc of what you want
    """

    date_arr = pd.DatetimeIndex(date_arr)
    if isinstance(attrs,str):
        return getattr(date_arr,attrs) if not dict_output else {attrs:getattr(date_arr,attrs)}
    else:
        return type(attrs)([getattr(date_arr,x) for x in attrs]) if not dict_output else {x:getattr(date_arr,x) for x in attrs} # cast output to same type as input


def np_datetime64_to_datetime(np_dt64):
    """
    Convert numpy datetime64 to native datetime objects
    Follows numpy datetime64 --> pandas timestamp (to_datetime) --> datetime datetime (to_pydatetime).
    see https://stackoverflow.com/a/21916253
    see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.to_pydatetime.html 
    """
    return pd.to_datetime(np_dt64).to_pydatetime()

def datetime_to_np_datetime64(datetime_datetime):
    """
    Convert datetime.datetime objects to np.datetime64 objects
    - isoformat converts to a datestring, which datetime64 can read
    -- isoformat doesn't seem to be vectorize, so added a loop check
    
    returns numpy ndarray if input is ndarray
    otherwise expects single values or iterable types
    """

    if isinstance(datetime_datetime, np.ndarray):
        return np.vectorize(lambda x: np.datetime64(x.isoformat()), otypes=['datetime64[ns]'])(datetime_datetime)
    elif personal.data_structures.isiterable(datetime_datetime):
        return np.datetime64(np.array([x.isoformat() for x in datetime_datetime]))
    else:
        return np.datetime64(datetime_datetime.isoformat()) # isoformat converts to a datestring, which datetime64 can read


def np_natural_add_timedelta(np_dt64, **relativedelta_args):
    """
    np_dt64 is a or an array of numpy datetime64 objects
    This can add things like months and years naturally to numpy datetimes, which don't typically work because they don't have set np.timedelta64 duration units like hours,minutes,seconds,etc do
    relativedelta_args are as https://dateutil.readthedocs.io/en/stable/relativedelta.html
    """
    d     = personal.date_operations.np_datetime64_to_datetime(np_dt64)
    delta = dateutil.relativedelta.relativedelta(**relativedelta_args)
    d     = d + delta
    return datetime_to_np_datetime64(d)
    
# def apply_fcn_by_grouping(A,grouping=None):
#     """ Grouping is hour, day, month, year, decade, century, millenia 
#     A is thus an xarray or pandas or similar object that permits grouping
#     The idea of this function is that it could work on a dataset that isn't dask equipped or something like that"""
    
def drop_leap_days(datetime_vec):
    date = pd.to_datetime(datetime_vec) # best way to access attributes like year month etc
    return datetime_vec[~((date.month == 2) & (date.day == 29))]
    
def xr_drop_leap_days(A,date_axis='time'): 
    """
    Drop all data from leap days from dataset with a valid datetime dateaxis
    """
    return A.sel({date_axis: ~((A[date_axis].dt.month == 2) & (A[date_axis].dt.day == 29))})
                 
       
def xr_groupby_strftime(A,date_coord='time',strfmt='%Y-%m-%d %H:%M:%S.%f',new_time_name=None):
    """
    see https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior for string formats
    """
    if isinstance(date_coord, int): # should be dataarray not dataset, for order
        d_type = type(A)
        if d_type is xr.core.dataarray.DataArray: # is xarray DataArray
            date_coord = A.dims[date_coord]
        elif d_type is xr.core.dataarray.Dataset: # is xarray DataArray
            data_variable = personal.data_structures.xr_variables_only(A) # pick first data variable, if this doesn't work you should have picked a name not a number
            data_variable = data_variable[list(data_variable.keys())[0]]
            date_coord = data_variable.dims[date_coord]
        

    indexer = A[date_coord].dt.strftime(strfmt)
    if new_time_name is not None:
        indexer = indexer.rename(new_time_name)
    out = A.groupby(indexer)


    return out
                 
def xr_groupby_date_of_year(A,date_coord='time',new_time_name='date'):
    """
    There is no great groupby 'dateofyear' option... `time.dayofyear' is ok, but does ordinal days after dec31 from previous year 
    ... see https://github.com/pydata/xarray/issues/1844
    """
    return xr_groupby_strftime(A,strfmt='%m-%d',new_time_name=new_time_name)             
     
def skip_past_leap_days(datetime_vec):
    """
    Given a time vector (presumably continuous), skips 1 day forward for the remainder of the vector at the first occurence of each leap day
    Would use contigous section detection to skip forward only once for each leap day...
    but each time you jump forward a day, your leap day moves up and we don't assume uniform spacing so we must redetect... 
    ... iterate until we reach the end of the vector...
    
    shouldn't fail unless your vector has 2 adjacent values from different leap days in which you could probably do someting else anyway...
    """
    
    date = pd.to_datetime(datetime_vec) # best way to access attributes like year month etc

    is_leap = ((date.month == 2) & (date.day == 29))
    while is_leap.nonzero()[0].any(): # while we still have leap values
        first_section_start = personal.data_structures.contiguous_regions(is_leap)[0][0]
        print('skipping leap day at ' + str(date[first_section_start]) + ' at index ' + str(first_section_start))
        date.values[first_section_start:] += pd.Timedelta('1 days')
        is_leap = ((date.month == 2) & (date.day == 29))
    return date.to_numpy()
    
    
                 
                 
                 
                 