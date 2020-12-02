import numpy as np
import xarray as xr
import personal.data_structures
from scipy import fft

def complete_fft():
    """
    *** Adapted from MATLAB Code ***
    
    Creates the one tailed FFT of a spectra and can correct the frequency axis...
    Note -- this returns time/length frequency, not angular frequency!
    sample_freq_or_period is a T/F but should be deprecated to just pick one -- don't make more complicated functions (i think freq is better?)
    """
    return
    
def complete_rfft():
    """
    same as above but with real fft (rfft)... maybe combine with a function method argument?
    """
    if dim_values is None:
        dim_values = list(range(data.shape[dim]))
    L = len(dim_values) #?????
    
    dim_values = dim_values - dim_values[0] # normalize to start from 0
    Ts         = dim_values[1] - dim_values[0]; Fs = 1/Ts; # sampling frequency and period 
    return
    

def climatology(data,dim=-1,harmonics=2,**kwargs):
    """
    *** Adopted from MATLAB Code ***
    Compute climatology from an input data set using real ffts (assumes real? idk if that's bad...)
    %  You are responsible for removing mean and trends separately if desired 
    %  You are responsible for ensuring even sampling frequency / interpolating
    %  - I would do it but hard w/ different calendars (gregorian, noleap, etc) and no nonstandard calendar in Python/xarray

    -- data      =   input data (assumes contains only a dataset that spans dim; could for example be a year or a mean of several years)
    -- dim       =   the dimension of the record dimension (name or int) along which we're taking fft
    -- harmonics =   the scalar # of harmonics you wish to retain (starting from length of input data) or an array of harmonics to retain
        - default of 2 is just enough to capture 0 (the mean) and 1 (some variability)
    """
    
    L = data.shape[dim]
    L_out = L//2 + 1 # bc is one sided out from rfft
    if np.isscalar(harmonics): # convert to list if needed
        if harmonics > L_out:
            raise ValueError(' Too many harmonics requested, maximum resolution for signal of length ' + str(L) + ' is ' + str(L) + '//2 + 1 = '  +  str(L_out) )
        harmonics = list(range(harmonics))
    
        
    # Calculate climatology (Filter to include only the first n-harmonics of variability (e.g. only keep seasonal variations) then invert
    # - Take Fourier transform; apply fft along dim_values dimension, returns 2-sided spectrum, divide by L to normalize coefficients;
    fftClimo = fft.rfft(data,n=L,axis=dim,**kwargs); #  n=next_fast_len(L) would pad trailing 0's for speed but meh
    # Truncate Fourier series
    truncFFT = fftClimo;
    
    trunc_inds = personal.data_structures.slice_select(data,dim=dim,ind=harmonics,return_indices=True)
    # is one sided so easy to cut out
    mask  = np.ones(truncFFT.shape,dtype=bool) # mask useful for inversion bc we wanna keep the selected values
    mask[trunc_inds] = False # values we wish to keep set to false
    truncFFT[mask] = 0 # set remaining True values to 0
    climo = fft.irfft(truncFFT,n=L,axis=dim,**kwargs); # think it's already real? will have to check...
    return climo
    
    
 #%% To do :: Think up another version to deal w/ NaN in data 
 #          - i.e. auto extrapolate before fft or other methods for missing data?)
 #          - or some least squares fit (should probably be separate option since extrapolating probably beter for 1 or 2 missed frames
 #            from many). Inclusion of the former would only be a wrapper, the second maybe a separate fcn bc that would need entirely
 #            separate fcn calls... Should create analagous version for those in scipy.fft for use in things like climatology() above... 
     

    
#      .rolling(time=time_smoothing[n],center=True,min_periods=1).mean()
def calculate_anomalies(data, axis=-1, t=None, baseline_methods=[[['mean']]],verbose=False,apply_ufunc_kwargs={},dask="allowed",dask_gufunc_kwargs=None):
    """
    You can remove (defaulting to acting along dimensions, but you can specify args):
        -- harmonic fit    # personal.math.climatology
        -- trend           # 
        -- mean            # np.mean
        -- rolling mean    # personal.math.move_mean() -- but consider, your mapping is supposed to tie back to the real data, so either apply along the entire array or in your groupings just take the normal mean first then apply rolling
        -- some custom fcn # 
    These could apply to some grouping mean:
        -- annual
        -- daily
        -- annual all
        -- daily all
        -- some grouping fcn
        
    Assumes underlying data types are numpy or dask arrays...
    
    e.g. baselines = [   [[('annual all', 'mean'), {'args':(),'kwargs':{}}],  [(None, 'rolling mean'), {'args':(),'kwargs':{'axis':0, 'window':8000, 'min_count':1}}] ], 
                         'rolling_mean'
                     ]
                     
                     would first calculate the annual mean of the entire dataset ('all', as opposed to removing each year's annual mean), then take its rolling mean of window length 5, and remove that result from the entire dataset...
                     ... followed by then removing a rolling mean from the resulting DataArray
                     
                     if you wanted to remove a mean say over multiple dimensions, you could construct your own fcn to do it
                     
        date_axis is optional but can indicate whether the desired axis is a date axis... ordinarily strings in time_selections default to using date operations for grouping... if date_axis is false they will not. Useful in case you had an non date dimension named daily or something like that
    """
    
#     def str_split_inclusive(string, delim): return [ _ + delim for _ in string.split(delim) ] # split while keeping the delimiter
    full_time = '%Y-%M-%d %H:%M:%S.%f'
    time_selections = { 'annual'     : '%Y'                   , # subtract from each year from that year
                        'annual all' :    '%M-%d %H:%M:%S.%f' , # subtract from all years from all years
                        'daily'      : '%Y-%M-%d'             , # subtract from each day from that day
                        'daily all'  :          '%H:%M:%S.%f' , # subtract from all days from all days
                      }
    
    method_functions = { 'harmonic fit' : climatology,
                         'trend'        : None,
                         'mean'         : lambda x,*args,**kwargs: np.mean(x,*args,keepdims=True,**kwargs),
                         'rolling mean' : personal.math.rolling_mean
                       }
    

    
    def print_verbose(x,verbose=verbose):
        if verbose: print(x)
        return
    
    if baseline_methods is None:
        print_verbose('No baseline method assigned, returning')
        return data
    
    if dask_gufunc_kwargs is None:
        dask_gufunc_kwargs = [{}]
    dask_gufunc_kwargs = dask_gufunc_kwargs * (len(baseline_methods)//len(dask_gufunc_kwargs)) # broadcast to same shape
    
    if apply_ufunc_kwargs is None:
        apply_ufunc_kwargs = [{}]
    apply_ufunc_kwargs = apply_ufunc_kwargs * (len(baseline_methods)//len(apply_ufunc_kwargs)) # broadcast to same shape
    
    d_type = type(data)
    if d_type is xr.core.dataarray.DataArray: # is xarray DataArray
        pass
    elif d_type is xr.core.dataarray.Dataset: # is xarray DataArray
        return data.map(lambda x : calculate_anomalies(x, axis=axis, baseline_methods=baseline_methods,verbose=verbose,dask=dask,dask_gufunc_kwargs=dask_gufunc_kwargs)) # just call the fcn on itself, mapped to variables (dataarrays) which are handled by above case
    else: # ideally is numpy or dask array, and we create a DataArray and assign this dim to it
        data = xr.DataArray(data)
        if t is None: t = np.arange(np.shape(data)[axis])
        data = data.rename({data.dims[axis]:'t'})         # rename that dimension
        data = data.assign_coords({'t':t})                # assign values for the dimension
        return calculate_anomalies(data, axis=axis, baseline_methods=baseline_methods,verbose=verbose,dask=dask,dask_gufunc_kwargs=dask_gufunc_kwargs).data # apply to dataarray, but return the data at the end since we didn't start with a dataarray


    
    if isinstance(axis, int):
        coord = data.dims[axis] # get name of dimension we're working on
    elif isinstance(axis,str):
        coord = axis
        axis  = data.get_axis_num(coord) # get the number of the dimension, same as  data.dims.index(coord) 
    
#     print_verbose(baseline_methods)
    for (i,baseline) in enumerate(baseline_methods):
        if not personal.data_structures.isiterable(baseline): baseline = [baseline] # make into a list if 
        print(baseline)
        for action in baseline:                                      # the grouping, e.g. [('annual all', 'mean'), {'args':(),'kwargs':{}}] or 'rolling_mean'
            print(action)
            baseline = data.copy()
            if isinstance(action,str): action = [action]             # -- for example for 'rolling mean' above
            if len(action) == 1: action = [action[0], {'args':(),'kwargs':{}}]# -- add in placeholder for args if we gave none, tuple and dict unpacking
            method = action[0]                                       # e.g. ('annual all', 'mean') or 'rolling_mean'
            print(action)
            if len(method) == 1: method = [None, method[0]]          # -- e.g. for 'rolling_mean' prepend None as the groupby
            method_grouping  = method[0]                             # e.g. 'annual all' or None
            method_fcn       = method[1]                             # e.g. 'mean' (handle the actual fcn later)
            if isinstance(method_fcn, str): method_fcn = method_functions[method_fcn] # convert string to fcn if necessary 
            
            if isinstance(method_grouping, str): # assume is a time grouping string
                if method_grouping in time_selections:
                    method_grouping = time_selections[method_grouping]
                    print_verbose("grouping data by time")
                    baseline = personal.data_structures.xr_groupby_strftime(baseline,date_coord=coord,strfmt=method_grouping,new_time_name=None)
                else:
                    baseline = baseline.groupby(method_grouping) # assume is the name of a dimension for example
                    
            elif callable(method_grouping): # is a fcn
                coord_vals = data['coord'].values
                baseline = baseline.groupby(method_grouping(coord_values))
                
            elif personal.data_structures.isiterable(method_grouping): # is some array or list or other such iterable:
                baseline = baseline.groupby(method_grouping)
            elif method_grouping is None: # group entire array
                pass #
            
            
            args   = action[1].get('args'  ,())
            kwargs = action[1].get('kwargs',{})
            if method_grouping is not None:
                print_verbose('applying baseline action: ' + str(action) + ' to data grouped by: ' + str(method_grouping) + '...')
                baseline = baseline.map(lambda x: method_fcn(data, *args, **kwargs)) # apply function to grouping, which returns data array
            else:
                print_verbose('applying baseline method: ' + str(action) + ' to data...')
#                 baseline = method_fcn(baseline, *args, **kwargs) # apply directly to dataarray
                if dask:
                    baseline = xr.apply_ufunc( lambda x:  method_fcn(x, *args, **kwargs), baseline,dask=dask,**apply_ufunc_kwargs[i])
                else:
                    baseline = xr.apply_ufunc( lambda x:  method_fcn(x, *args, **kwargs), baseline,dask=dask,dask_gufunc_kwargs=dask_gufunc_kwargs[i],**apply_ufunc_kwargs[i])
                    
            print_verbose('...done')
        
        
                

#     print((data, baseline))
#     return (data - baseline, baseline)
    return data - baseline # can only return this one thing beecause of recursive nature for other data types
#     return (baseline)
#     return baseline
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    