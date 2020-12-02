# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:19:48 2020

@author: Jordan
"""

# In this we opt for putting *args before the positional arguments so we can call the fcn without having to leave all the names out bc of a dx or dy or something at the end 
# I.e. now you can call grad(x,dx,dy,method='npgradient',...)... with *args first
# where as grad(x,method='npgradient',dx,dy,...) would not have been valid with *args at the back, only grad(x,'npgradient',...,dx,...) bc positional args cannot follow keyword args



# issues    - gradient wants x,y grids, not dx dy
#           - diff doesn't handle spacing but i guess that could be handled externally?
#               - we don't want dx, dy, dz, dt, ... to be default args bc of the unknown number of dimensions


import numpy as np
import personal.data_structures
import personal.function_manipulation
import itertools
import xarray as xr
import bottleneck
import scipy.signal



def dq_to_q(dq,dim=-1):
    """
    Taking dq, which varies along dim, create the q matrix by starting from 0 and doing cumsum along the dim
    """
    # prepend 0 and cumsum
    return np.cumsum(personal.data_structures.pad(dq,dim=dim,num_l=1,num_r=0,mode='constant',constant_values=0), axis=dim)

    
def periodic_wrapper(A,dq_dim,n=1): # maybe mve back to math?
    """
    Takes arrays in varargs and pads them by repeating last/first n values and rpeating them before/after the array along dimension
    Assumes A is the array, and vardqs are possible coordinates (useful for handling uneven spacing, e.g. from lat/lon --> x,y data)
    Returns list of arrays (to run on multiple arrays use list comprehension)
    
    dq_dim is a dict mapping the dq of each coordinate to the dim it represents
    returns dictionary of form {dim: [A__padded_on_dim, pseudo_q_dim__padded_on_dim]}
    pseudo_q_dim is created from the padde dq values, set to 0 at index 0 of 
    """
    
    # We don't preallocate, probably not worth for list comprehension
    # pads A along dims, and also dq but converts dq to q so something like npgradient could use
    return {dim: [personal.data_structures.pad(A,dim,num_l=n,num_r=n,mode='wrap'),dq_to_q(personal.data_structures.pad(dq_dim[dim],dim,num_l=n,num_r=n,mode='wrap'),dim)] for dim in dq_dim.keys()}




def np_gradientn(A,n,*varargs,**kwargs):
    """ applies numpy.gradient() n times to  calculate the nth gradient """
    return personal.function_manipulation.repeat(np.gradient,A,n,*varargs,**kwargs)


def grad(A,*var_qs, method='npgradient',periodic=False, axis = None,n=1,**kwargs):
    """ 
    Calculates the gradients along axes in axis, if axis is None, returns gradients in all dimesions of A
    Assumes *var_qs holds:
       (both)          a constant scalar spacing between array elements
       (both)          or N scalar spacings for the N values in axis (or all axes in A if axis is None)
       (npgradient)    or in order, the coordinates (not differntials) for the dimensions in axis (if none passed, assumes all spacing is 1, if scalar assumes same scalar for all axes)
    Assumes periodic holds the axes in axis that are periodic, if is False, assumes no axes are periodic. If axis is a scalar, so should periodic

    # methods - 'npgradient', 'npdiff' from numpy.gradient, numpy.diff
       - npgradient is very general with spacing etc
       - npdiff only should be used with equal spacing if it is to be a real derivative
       - TO ADD (spline and other interpolative methods, esp for irregular data)

    n = nth derivative

    ******************************
    Note periodic boundary conditions can only exist with evenly spaced data (otherwise padding x,dx is impossible without the looping (end:0) distance known)
       - if not even spacing, one should do padding and calculation of dx outside this fcn and calculate a new pseduo-x to be passed in and gradient taken without periodic flag
           -- for periodic lat/lon data, one could write a wrapper fcn to do this 
    ******************************

    return list of arrays/ndarrays if axis is iterable, returns just the output array if axis is scalar
    """
    
    if method not in ['npgradient','npdiff']:
        raise ValueError('Method must be npgradient or npdiff')

    nd = np.ndim(A)
    sz = np.shape(A)

    var_qs = list(var_qs)
    
#     print((' oi lookee here',[x.shape for x in var_qs]))

    #print(A.shape)

    return_scalar = False # default to not returning just the array
    if axis is None:
        axis = tuple(np.arange(nd)) # operate over all axes, np.gradient takes in tuple
    elif np.isscalar(axis):
        return_scalar = True # return just the array if axis was a scalar (only one derivative/gradient)
        axis = np.array([axis])
    # else, axis is a list of axes
    periodic = np.broadcast_to(periodic,np.shape(axis)) # Ensures periodic is of same size as axes (e.g. if it was single logical, broadcasts to array of logical matching length of axes)

    #print(var_qs)
    #print(axis)

    # Handle spacing/periodic relations
    la = len(axis)
    nv = len(var_qs)
    if nv not in [0,1,la]:
        raise ValueError('var_qs must contain either a single scalar value a combination of scalars and coordinate arrays to match the differentiated dimensions')
    if nv == 0:                                      # no spacings passed in
        var_qs = np.ones(la)                            # spacing 1 for all axes
    elif nv == 1:                                    # 1 spacing passed in, so either len(axis) is 1 (only one gradient to take) or same constant spacing for all axes
        if la != 1:                                      # should be a scalar spacing if more than one axis is being processed but only one spacing provided
            if np.isscalar(var_qs[0]):                        # is scalar spacing
                var_qs = var_qs[0]*np.ones(la)                    # scalar spacing from var_qs[0] value for all axes
            else:                                            # is not scalar spacing but should be
                raise ValueError('var_qs should represent a scalar spacing if only one value given')
        else:                                                # only one axis was given so var_qs could be a scalar or a vector or a full coordinate array
            pass
    else:                                           # we do have a spacings given for each axis, a combination of scalars and vectors and coordinate arrays
        pass

    
    
    #Handle Padding Values
    pad_npg = dict()
    for i,ax in enumerate(axis):
        pad_npg[ax] = periodic[i] * n # pad n if periodic
       #pad['npdiff'    ][ax] =   **** FILL IN LATER ***

    # Handle periodicity for var_qs (Only need to periodicize if var_qs[ax] is a coordinate array, for scalars can just periodize A )
    for i,ax in enumerate(axis):     
        if np.isscalar(var_qs[i]):  # we received a single scalar - aka a spacing value
            if method == 'npgradient':
                pass # npgradient accepts spacing scalars
            elif method == 'npdiff':
                 pass         # ******* FILL IN LATER *******
        elif  np.ndim(var_qs[i]) == 1:    # is a 1D coordinate array, gets periodized normally, by only padding
            if method == 'npgradient':
                var_qs[i] = personal.data_structures.pad(var_qs[i],num_l=pad_npg[ax],num_r=pad_npg[ax],mode='wrap') # just pad normally 
            elif method == 'npdiff': 
                pass        #  ******* FILL IN LATER *******
        else: # is array
            if method == 'npgradient':
                if personal.data_structures.is_equal_along_dimension(var_qs[i],ax): # check if array is all the same along the axis ax in question 
                    
                    # deprecate this, this is bound to be slow, let the user do it 
                    #dim_inds  = {i:0 for i in range(la) if i != ax} # slice along dimension ax (leaving out ax means we slice everything), no lists means drop dimensions so we'll get 1D array
                    #var_qs[i] = personal.data_structures.any_select(var_qs[i],dim_inds) #(if it is i.e. a pure grid input we can just run npgradient with a 1d coord vector)
                    var_qs[i] = personal.data_structures.pad(var_qs[i],num_l=pad_npg[ax],num_r=pad_npg[ax],mode='wrap') # just pad vector normally 
                else:
                    var_qs[i] = personal.data_structures.pad(var_qs[i],dim=ax,num_l=pad_npg[ax],num_r=pad_npg[ax],mode='wrap') # we have unstructured data so we'll need the full padded q field and will have to run gradient on it along dimension
            elif method == 'npdiff':
                pass        #  ******* FILL IN LATER *******



   # Pre- allocate (does this do anything? I think so)
    if method == 'npgradient':
        out = [np.full(sz, np.nan) for ax in axis]  # there is no size reduction, periodic or not, since np.gradient does edge estimation
    elif method == 'npdiff':
        out = [np.full(np.subtract(sz,personal.data_structures.replace(np.zeros(nd),ax,n*(1-periodic[ax])).astype(int)), np.nan) for ax in axis] # shrinks dimensions, if is periodic subracts nothing, if isn't subtracts 1 for each diff
    else: # implement nproll (probbly only first and second derivatives?) or by convolution
        pass

   # Handle Differentiation
    for i,ax in enumerate(axis): # the list of dimensions we take gradients in
        if method == 'npgradient':
            AA = personal.data_structures.pad(A,dim=ax,num_l=pad_npg[ax],num_r=pad_npg[ax],mode='wrap') # wrap A along dimension
            if np.isscalar(var_qs[i]) or np.ndim(var_qs[i]) == 1: # scalar or vector spacing
                AA = np_gradientn(AA,n,var_qs[i],axis=ax,**kwargs) # differentiate
            else: # full array spacing indicating we had uneven gridding
                # --------------------------------------------------------------------- #
                  ## deprecated, clean up soon, can delete
                    
#                 do_n = lambda A, dq : np_gradientn(A,n,dq,axis=0,**kwargs) # abstract away the n, gets fed A and dq from apply_along_axis (so axis 0), inserts n in between and calls gradientn. Not sure if kwargs goes through successfully but i think so
#                 AA = personal.data_structures.apply_along_axis(do_n, ax, [A,var_qs[i]]) # axis = 0 automatically bc gradients taken between 2 1D arrays
                
                # --------------------------------------------------------------------- #
                sz_q = np.shape(var_qs[i])
                protected_dims = [i for i in range(len(sz)) if sz[i] != sz_q[i]]
                # since we have protected dims, we may get out dq that are not 1D, and then we should squeeze, they should be vectors, just perhaps not in 1D
                def do_n(A,dq,axis=len(protected_dims)): # the axis to work on would be 1 after the number of protected dims, but 0 indexed so just len() works
                    out = np_gradientn(A,n,dq.squeeze(),axis=axis,**kwargs)
                    return out # abstract away the n, gets fed A and dq from apply_along_axis (so axis 0), inserts n in between and calls gradientn. Not sure if kwargs goes through successfully but i think so
                AA = personal.data_structures.apply_along_axis(do_n, ax, [A,var_qs[i]], protected_dims=protected_dims) # axis = 0 automatically bc gradients taken between 2 1D arrays
                # -------------------------------------------------------------------- #
            out[i] = personal.data_structures.slice_select(AA, ax,range(pad_npg[ax],np.shape(AA)[ax]-pad_npg[ax])) # trim  first and last n that we added (npgradient maintains size)
        elif method == 'npdiff':
            pass #  ******* FILL IN LATER *******

    # Handle Return
    return out if not return_scalar else out[0] # if axis was a scalar passed in, just return the output array


       
       # for i,ax in enumerate(axis): # axis is a list of dimensions we take gradients in
       #     if method == 'npgradient': # use numpy gradient
       #         if periodic[i]: # pad on both sides of axis for nth differencing, then cut ends (so no edge estimation), padding is n on both sides
       #             AA = personal.data_structures.periodic_pad(A,dim=ax,num_l=n,num_r=n) # pads
       #             AA = np_gradientn(xx,n,axis=ax,varargs[i],**kwargs) # calculate the gradient using the coordinates 
       #             out[i] = personal.data_structures.slice_select(AA, ax,range(n,np.shape(xx)[ax]-n)) # cut of first and last n that we added (npgradient maintains size)
       #         else: # not periodic, we still have to separate by axis so we can apply an arbitrary n number of times
       #             out[i] = np_gradientn(x,n,axis=ax,*varargs,**kwargs)
                   
       #     elif method == 'npdiff': # use numpy diff
       #         if periodic[i]: # pad on both sides of axis for nth order differencing int(np.floor(n/2)) on left and int(np.ceil(n/2)) on right
       #             l_ind = int(np.floor(n/2))
       #             r_ind = int(np.ceil(n/2))
       #             AA = personal.data_structures.periodic_pad(A,dim=ax,num_l=l_ind,num_r=r_ind)
       #             xx = np.concatenate((personal.data_structures.slice_select(x,ax,range(-l_ind,0,1)), x,personal.data_structures.slice_select(x,ax,range(0,r_ind,1))),axis=ax)
       #             out[i] = np.diff(xx,n,axis=ax,*varargs,**kwargs) # diff did the cutting for us
       #         else:
       #             out[i] = np.diff(x,n,axis=ax,*varargs,**kwargs)
       # return out if not return_scalar else out[0] # if axis was a scalar passed in, just return the output array

    # magnitude of the gradient, square each array in output and then sum ([x**2 for x in out]), use periodic or gradientn
  
   
def curl(F,dims_to_space_dims=[],coord_arrays={0:1,1:1,2:1},method='npgradient',periodic=False,dims={'x':0,'y':1,'z':2},**kwargs):
    """ Takes the curl of vector field F, F represented by a list of length n=2,3 providing the magnitude of each of its components at x,y,(z), i.e. in 2d U,V , in 3D U,V,W
        The second two options tell grad how to take derivatives
        
        If only 2 components of F are provided (e.g. U,V), takes the curl assuming these components live in the first 2 (x,y) dimensions of 3D, but only returns the 3rd/z-component of the curl
        Assumes axes are ordered 0-x, 1-y, 2-z 
        
        If you only pass in 3D arrays U,V you'll       only get the z component of the curl but it will be a 3D field (scalar value at x,y,z)
        If you only pass in 2D arrays u,v you'll still only get the z-component of the curl  so it will be a 2D field (scalar value at x,y)
        
        Args like coordinate spacing/location can be passed via varargs/kwargs -- see documentation for differentiation method for correct syntax.
        Since each component of F is a full array in all dimensions, there should be no issue"""
        
        # dims_to_space_dims would map array dimensions to spatial dimensions
        # youd wanna provide coords that match the format needed for your gradient method
    
#     for dim,arr in F.items():
#         print(arr.size)
#         if arr.size ==0:
#             print('shirt circuiting')
#             return [np.array([0]) for dim in F.keys()]
    
    curl = {}     
    # curl is cyclical so from i, going backwards is (i-1 mod 3), (i-2 mod 3)
    for i in [1,2,3]:
        if np.mod(i-1,3) in F.keys() and np.mod(i-2,3) in F.keys(): # then curl[i] exists
            a = np.mod(i-1,3)
            b = np.mod(i-2,3)
            curl[i] = grad(F[a],coord_arrays.get(b,1),method=method,periodic=periodic,axis=dims_to_space_dims.get(b,b),n=1,**kwargs) - grad(F[b],coord_arrays.get(a,1),method=method,periodic=periodic,axis=dims_to_space_dims.get(a,a),n=1,**kwargs)
    
    return curl


def div(F,method='npgradient',periodic=False,*varargs,**kwargs):
    """ Like curl fcn but is div
    You only need to provide as many components in F as you want (and could give more than 3)
    Assumes components are in right-hand rule order (x,y,z,...), truncated if fewer components are provided
    
    Returns a scalar so is just the array"""
    
    nd = len(F) #number of dimensions, should match np.ndim() of the individual components of F (F_x,F_y,(F_z))
    
    S = grad(F[0],method=method,periodic=periodic,axis=0,*varargs,**kwargs) # scalar field
    for i in range(1,nd):
        S  += grad(F[i],method=method,periodic=periodic,axis=i,*varargs,**kwargs) # add other components as provided
    return S    
    
    
       
   
    
def Jacobian(F,deriv_dims=[],coord_arrays={0:1,1:1,2:1},method='npgradient',periodic=False,**kwargs):
    """ Takes the Jacobian of vector field F, F represented by a list of length n providing the magnitude of each of its components at x,y,(z), i.e. in 2d U,V , in 3D U,V,W
        The second two options tell grad how to take derivatives
        
        coord_arrays provides the spacing arrays for the coordinates in F 
        deriv_dims provides the dimensions to include in the Jacobian, default to all i.e. [1:len(F)]
        
        Takes derivative in the form pdv 
        
        Args like coordinate spacing/location can be passed via varargs/kwargs -- see documentation for differentiation method for correct syntax.
        Since each component of F is a full array in all dimensions, there should be no issue"""
        

        # dims_to_space_dims would map array dimensions to spatial dimensions
        # youd wanna provide coords that match the format needed for your gradient method
        
    #nd = len(F) # num of dims we are iterating over (i.e. those provided in F).    
    if not deriv_dims: # list is empty, default to all
        nd2 = np.ndim(F[list(F.keys())[0]]) # get dimensionality of arrays from the inherent sub arrays. Dimensions such as time could be included in nd2 below that are not in nd
        deriv_dims = list(range(nd2))
        
    coord_arrays_list = [coord_arrays[i] for i in deriv_dims] # get ordred list

    J = {}
    for i in F.keys(): # iterate over components of F
        J[i] = {}
        out = grad(F[i],*coord_arrays_list,method=method,periodic=periodic,axis=deriv_dims,**kwargs) # returns list
        for j,ax in enumerate(deriv_dims):
            J[i][ax] = out[j] # output returns only as list (maybe change to dict later and propagate? idk)
    return J

def combinations(a, r, allow_repeats = False):
    """
    Return successive r-length combinations of elements in the array a.
    This operates along dimenssion 0 of array a for itertools (allow_repeats = False)
        > Should produce the same output as array(list(itertools.combinations(a, r))), but faster.
    For allow_repeats = True (uses np.meshgrid), you have to supply only 1-D arrays
    
    allow_repeats=False means you get no repeated values in the outputs, and doesn't repeat with different orders (order doesn't matter)
    """
    if allow_repeats:
        return np.flip(np.array(np.meshgrid(*((a,)*r),indexing='ij')).T.reshape(-1,r),axis=1)
    else:
        a = np.asarray(a)
        dt = np.dtype([('', a.dtype)]*r)
        b = np.fromiter(itertools.combinations(a, r), dt)
        return b.view(a.dtype).reshape(-1, r)
    
def geometric_mean(arr,axis=None,omitna=False,**kwargs):
    """
    Coded with logs first to avoid overflow.
    Takes the geometric mean along any axis...
    """
    
    if omitna:
        mean_fcn = np.nanmean
    else:
        mean_fcn = np.mean
        
    return np.exp(mean_fcn(np.log(arr), axis=axis, **kwargs))
#     return np.exp(arr.sum()/len(a))
    
    
def range_bounds(arr, axis=None,omitna=True,stack_dim=None, **kwargs):
    """
    Returns the min and max of data along dim at the same time
    """
    if omitna:
        max_fcn = np.nanmax
        min_fcn = np.nanmin
    else:
        max_fcn = np.max
        min_fcn = np.min
        
    if stack_dim is None:
        if axis is None:
            stack_dim = -1           # if takes across all axis, just stack on last axis
        else:
            stack_dim = np.max(axis) - len(axis) + 1 # default to latest axis used in the calc, but keep in same position relative to other axes
    

    return np.stack((min_fcn(arr,axis,**kwargs),max_fcn(arr,axis,**kwargs)),axis=stack_dim)

def rolling_mean(arr,axis=-1,window=1,min_count=None):
    """ 
    Meant to work on ndarrays since numpy and pandas thought it ill posed to allow this...
    Using bottelenck rn but it also glosses over the nans in the interior which may not be desired behavior... think of way to omitna or not later...
    """
    
    return bottleneck.move_mean(arr, window, min_count=min_count, axis= axis)

def monotonicize(arr,axis=-1,direction='increasing'):
    """
    makes an array monotonic along axis by repeating values as necessary to keep an increasing or decreasing trend
    
    If you only want one instance of each value, call unique via  _, idx = np.unique(arr, return_index=True), arr[np.sort(idx)] on output of this function
    """
    
    if direction == 'increasing':
        return np.maximum.accumulate(arr)
    elif direction == 'decreasing':
        return np.minimum.accumulate(arr)
    else:
        raise ValueError('Unknown direction ' + str(direction) + ', direction must be ''increasing'' or ''decreasing''')
        
        


def random_with_clustering(nmax=100, L=1, mean_cluster_size=1, cluster_size_distribution_fcn= lambda mcs,n: np.maximum(1,np.round(np.random.normal(scale=mcs**(1./2), size=n)) + mcs),safety_counter_max=1e4):
    """
    Makes random clusters with adjacency clustering using integers in range (0...nmax)
    nmax = maximum value (int)
    L    = final output length (int), L < nmax
    
    Process is:
    - generates n = round(L / mean_cluster_size) cluster sizes with random size centered around mean_cluster_size using cluster_size_distribution_fcn(n, center=mean_cluster_size, range=[0..nmax])
    - selects   n unique random cluster centroids
    - merges indices to only the unique subset from all the clusters
    -- if we have selected more than L indices, chooses a random set to discard
    -- "  "  "    "        less than L indices, chooses a random set to add
    --> The adding is probably fine, but the discarding might be less than ideal and reduce mean cluster size... but you can play around with it or implement a better method later...
    actually if we're over so we'll decrease our cluster sizes and then adding to correct for overlap can only help...
    
    """
    
    if L > nmax:
        raise ValueError('The number of final indices L = '+str(L)+' must be less than of equal to nmax = '+str(nmax))
    if L < 1:
        raise ValueError('L must be at least 1, you provided L = '+str(L))
    
    n = max(1,round(L / mean_cluster_size)) 
    cluster_sizes = cluster_size_distribution_fcn(mean_cluster_size, n).astype(int)
    adjustment_size = L - sum(cluster_sizes)
    safety_counter = 0
    while adjustment_size != 0:
        to_adjust = np.random.default_rng().choice(n,size=min(n,abs(adjustment_size)), replace=False) # choose random indices to adjust
        if adjustment_size > 0: # we need to add adjustment_size to our clusters
            for i in to_adjust:
                cluster_sizes[i] +=1
        elif adjustment_size < 0:
            for i in to_adjust:
                if cluster_sizes[i] !=  1: # don't destroy clusters (setting cluster size for that cluster to 0 -- I don't think this can spin infinite loops unless mean_cluster_size < 1)
                    cluster_sizes[i] -= 1
        else: # adjustment size is 0
            pass
        adjustment_size = L - sum(cluster_sizes)
        safety_counter += 1
        if safety_counter > safety_counter_max:
            raise ValueError('Did not converge to coherent cluster sizes... please check setup')

    
    cluster_centers = np.random.default_rng().choice(nmax,size=n, replace=False) # choose random clusters w/o replacement

    cluster_inds = [None]*n
    for i in range(n):
        sz      = cluster_sizes[i]
        c       = cluster_centers[i]
        span  = sz // 2
        is_even = 1 - (sz % 2)
        start   = c - span + is_even
        end     = c + span + 1
        if start < 0:
            # print('adjusting from start', nmax, (start,end))
            end   -= start # change end before start so doesnt get adjustied
            start -= start
        if end > nmax:
            # print('adjusting from end', nmax, (start,end))
            start -= (end-nmax) # change start before end so doesnt choke
            end   -= (end-nmax) # pin back to nmax, whih isn't included in range so works since nmax-1 is the last valid index
        cluster_inds[i] = np.arange(start, end) # should lean forward
        
    cluster_inds = np.unique(np.concatenate(cluster_inds)) #.sort(kind='mergesort')
    cluster_inds.sort(kind='mergesort') # sorts, returns None so just do it here
    
    to_add = L - len(cluster_inds) # is necessarily positive or 0 since removal is only by overlap and np.unique
    if to_add > 0:
        possible_additions = np.setdiff1d(np.arange(nmax), cluster_inds)
        additions = np.random.choice(possible_additions,to_add,replace=False)
        additions.sort() # sort so when they get inserted at insertion points in cluster_inds, they also remain sorted themselves
        insertion_points = np.searchsorted(cluster_inds, additions)
        cluster_inds = np.insert(cluster_inds, insertion_points, additions, axis=None)
        pass
        
    
    return cluster_inds
    
    
    
def symlog_rolling_mean(data, t=None, axis=-1,window=1, min_periods=1, center=True, keep_attrs=None, **window_kwargs):
    """
    uses xarray to do the mean...
    """
    
    if min_periods > window:
        raise ValueError('min_periods (' + str(min_periods) + ') cannot be greater than window (' + str(window) + ')')

    def nan2zero(x): return x.where(~np.isnan(x),0)
    
    d_type = type(data)
    if d_type is xr.core.dataarray.DataArray: # is xarray DataArray
        pass
    elif d_type is xr.core.dataarray.Dataset: # is xarray DataArray
        pass
    else: # ideally is numpy or dask array, and we create a DataArray and assign this dim to it
        data = xr.DataArray(data)
        if t is None: t = np.arange(np.shape(data)[axis])
        data = data.rename({data.dims[axis]:'t'})         # rename that dimension
        data = data.assign_coords({'t':t})                # assign values for the dimension
        return symlog_rolling_mean(data, t=t, axis=axis, window=window, min_periods=min_periods, center=center, keep_attrs=keep_attrs, **window_kwargs).data # apply to dataarray, but return the data at the end since we didn't start with a dataarray
    
    if isinstance(axis, int):
        coord = data.dims[axis] # get name of dimension we're working on
    elif isinstance(axis,str):
        coord = axis
        axis  = data.get_axis_num(coord) # get the number of the dimension, same as  data.dims.index(coord) 
    
    data = data.copy()
    
    pos = data.where(data >= 0, np.nan)
    neg = data.where(data <  0, np.nan);
    
#     print(data,pos,neg)
        
    pos = np.log10(pos) # will always be real numbers
    neg = np.log10(np.abs(neg))
    
#     print(pos,neg)
        
    pos_roll  = pos.rolling({coord:window}, min_periods=1,center=center, keep_attrs=keep_attrs, **window_kwargs)
    pos_count = pos_roll.count() # count non-nan values in this roll... use min_periods=1 so we never fail
    pos_count = nan2zero(pos_count) #pos_count.where(~np.isnan(pos_count),0)
    pos_mean  = pos_roll.mean()
    neg_roll  = neg.rolling({coord:window}, min_periods=1,center=center, keep_attrs=keep_attrs, **window_kwargs)
    neg_count = neg_roll.count()
    neg_count = nan2zero(neg_count) #neg_count.where(~np.isnan(neg_count),0)
#     print(neg_count)

    neg_mean  = neg_roll.mean()
    
#     print(pos_mean, neg_mean)
    
    # unsure if to combine the log or raw values after taking means
    
    total      = pos_count + neg_count # w/ the zeros included it means that 
    neg_weight = neg_count / total
    pos_weight = pos_count / total
    

#     print(pos_weight,neg_weight)
#     print(neg_weight + pos_weight)
    
#     out = pos_weight*(10**pos_mean) - neg_weight*(10**neg_mean) # geometric weight?
    out = nan2zero(pos_weight*(10**pos_mean)) - nan2zero(neg_weight*(10**neg_mean)) # geometric weight?
    out[total<min_periods] = np.nan # we used nan2zero to allow combination (so pos or neg with no weight/values count as none but don't screw up the sum so we fix that here by enforcing our total count condition
    
    return out
    
    
def periodicize(x,q, dq0 = None,dim=-1, cut_middle=True):
    """
    repeats the values in x along dimension dim, and assigns new coordinates to them, then can the middle out and returns that so you have a 'periodic' signal
    Note, dq0 is the first value in dq (i.e. representing a shift from the end->beginning of x). Default is to use dq0 = q[0]-0, but if q[0] is 0, you need to supply dq0
    """
    
    dq = np.diff(q,axis=dim)
    
    if dq0 is None:
        if q[0] == 0:
            raise ValueError('Unsure how to specify coordinate distance for repeating signal, signal starts at q[0]=0, but the default is to use dq[0] = q[0] - 0')
        else:
            dq0 = q[[0]]

    dq = np.append(dq0,dq,axis=dim)
    dq = np.cumsum(dq,axis=dim)
    
    q2 = q[-1] + dq
    
    x_periodic = np.concatenate((x, x),axis=dim)
    q_periodic = np.concatenate((q,q2),axis=dim)
    
    if cut_middle:
        L = np.shape(x)[dim]
        cut = L//2
        if L%2 == 0: # even
            cut_beg = cut
            cut_end = cut
            # only gotta cut same off beginning and end
        else: # let's cut one more off end than beginning
            cut_beg = cut
            cut_end = cut+1
        x_periodic = x_periodic[cut_beg:-cut_end]
        q_periodic = q_periodic[cut_beg:-cut_end]

    return x_periodic, q_periodic    
    

    
    
def geometric_normalize(arr, axis=-1, order=2):
    """
    Normalizes vectors in arr along axis to be unit vectors (lenth 1)
    """
    magnitude_a = np.linalg.norm(arr, ord=order, axis=axis,keepdims=True)
    magnitude_a[magnitude_a==0] = 1 # avoid divide by 0 errors, 0/1 will still be 0
    return arr / magnitude_a



def extreme_extremum_along_axis(x, axis = -1, extremum = 'earliest max', return_full_inds_and_value=True, early_late_mode='wrap' ):
    """ options are earliest/latest max/min, and largest/smallest max/min
        
        - return_full_inds_and_value decides if we return the full ND indices as well as the value we calculated in this fcn
        - early_late_mode decides for argrelextrema whether or not we consider the ends to be possible maxima, default is yes (wrap back to beginning), alternate would be 'clip', see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelextrema.html
        
        This function may fail on all nan slices along axis... since it has a hard time applying the extreme_fcn it calculates and gettinga straight answer
    """
    
    qualifier, extreme = extremum.split(sep = ' ') # split our divider in 2
    if   any(map(qualifier.lower().__contains__, ['earl'])):
        def qualifier_fcn(x, extreme_fcn): return scipy.signal.argrelextrema(x, extreme_fcn, axis=0, mode=early_late_mode)[0][ 0]
        def resolution_fcn(inds, axis=axis):
            inds                   = np.expand_dims(inds, axis=axis) # add back the dimension that the qualifier would have removed, so should have shape of the original data, but with shape 1 along axis
            vals                   = np.take_along_axis(x, inds, axis=axis) # take the values corresponding to the inds we calculated, so should have shape of the original data, but with shape 1 along axis
            extreme_inds           = np.nanargmin(inds, axis=None) # we want the smalleset/earliest ind of all the largest inds along axis
            extreme_inds           = np.unravel_index(extreme_inds, vals.shape) # expand this back out to original shape of vals, so index is 0 on axis=axis which has length 1
            extreme_val            = vals[extreme_inds] # select the extreme value along the axis
            extreme_ind_along_axis = inds[extreme_inds] # get the actual ind along axis that we want to return to the user
            extreme_inds           = personal.data_structures.tuple_replace(extreme_inds,extreme_ind_along_axis, axis) # replace the index on axis with this value so that we can actually return a useful tuple to the user
            return extreme_ind_along_axis, extreme_val, extreme_inds if return_full_inds_and_value else extreme_ind_along_axis
        
    elif any(map(qualifier.lower().__contains__, ['late'])):
        def qualifier_fcn(x, extreme_fcn): return scipy.signal.argrelextrema(x, extreme_fcn, axis=0, mode=early_late_mode)[0][-1]
        def resolution_fcn(inds,axis=axis):
            inds                   = np.expand_dims(inds, axis=axis) # add back the dimension that the qualifier would have removed, so should have shape of the original data, but with shape 1 along axis
            vals                   = np.take_along_axis(x, inds, axis=axis) # take the values corresponding to the inds we calculated, so should have shape of the original data, but with shape 1 along axis
            extreme_inds           = np.nanargmax(inds, axis=None) # we want the largest/latest ind of all the largest inds along axis
            extreme_inds           = np.unravel_index(extreme_inds, vals.shape) # expand this back out to original shape of vals, so index is 0 on axis=axis which has length 1
            extreme_val            = vals[extreme_inds] # select the extreme value along the axis
            extreme_ind_along_axis = inds[extreme_inds] # get the actual ind along axis that we want to return to the user
            extreme_inds           = personal.data_structures.tuple_replace(extreme_inds,extreme_ind_along_axis, axis) # replace the index on axis with this value so that we can actually return a useful tuple to the user
            return extreme_ind_along_axis, extreme_val, extreme_inds if return_full_inds_and_value else extreme_ind_along_axis
        
    elif any(map(qualifier.lower().__contains__, ['large'])):
        def qualifier_fcn(x, extreme_fcn): return extreme_fcn(x, axis=0)
        def resolution_fcn(inds,axis=axis): # inds should be an array with 1 value for each vector along axis so that axis squeezed out.
            inds                   = np.expand_dims(inds, axis=axis) # add back the dimension that the qualifier would have removed, so should have shape of the original data, but with shape 1 along axis
            vals                   = np.take_along_axis(x, inds, axis=axis) # take the values corresponding to the inds we calculated, so should have shape of the original data, but with shape 1 along axis
            extreme_inds           = np.nanargmax(vals, axis=None) # we want the largest value of all the largest values along axis
            extreme_inds           = np.unravel_index(extreme_inds, vals.shape) # expand this back out to original shape of vals, so index is 0 on axis=axis which has length 1
            extreme_val            = vals[extreme_inds] # select the extreme value along the axis
            extreme_ind_along_axis = inds[extreme_inds] # get the actual ind along axis that we want to return to the user
            extreme_inds           = personal.data_structures.tuple_replace(extreme_inds,extreme_ind_along_axis, axis) # replace the index on axis with this value so that we can actually return a useful tuple to the user
            return extreme_ind_along_axis, extreme_val, extreme_inds if return_full_inds_and_value else extreme_ind_along_axis
        
    elif any(map(qualifier.lower().__contains__, ['small'])):
        def qualifier_fcn(x, extreme_fcn): return extreme_fcn(x, axis=0)
        def resolution_fcn(inds,axis=axis):
            inds                   = np.expand_dims(inds, axis=axis) # add back the dimension that the qualifier would have removed, so should have shape of the original data, but with shape 1 along axis
            vals                   = np.take_along_axis(x, inds, axis=axis) # take the values corresponding to the inds we calculated, so should have shape of the original data, but with shape 1 along axis
            extreme_inds           = np.nanargmin(vals, axis=None) # we want the smallest value of all the largest values along axis
            extreme_inds           = np.unravel_index(extreme_inds, vals.shape) # expand this back out to original shape of vals, so index is 0 on axis=axis which has length 1
            extreme_val            = vals[extreme_inds] # select the extreme value along the axis
            extreme_ind_along_axis = inds[extreme_inds] # get the actual ind along axis that we want to return to the user
            extreme_inds           = personal.data_structures.tuple_replace(extreme_inds,extreme_ind_along_axis, axis) # replace the index on axis with this value so that we can actually return a useful tuple to the user
            return extreme_ind_along_axis, extreme_val, extreme_inds if return_full_inds_and_value else extreme_ind_along_axis
    else:
        pass


    if    any(map(extreme.lower().__contains__, ['max'])):
            if   any(map(qualifier.lower().__contains__, ['earl','late'])):
                extreme_fcn = np.greater
            elif any(map(qualifier.lower().__contains__, ['small','large'])):
                extreme_fcn = np.nanargmax
            pass
    elif  any(map(extreme.lower().__contains__, ['min'])):
            if   any(map(qualifier.lower().__contains__, ['earl','late'])):
                extreme_fcn = np.less
            elif any(map(qualifier.lower().__contains__, ['small','large'])):
                extreme_fcn = np.nanargmin
            else:
                pass
            
    inds =   np.apply_along_axis(lambda x: qualifier_fcn(x,extreme_fcn) , axis, x )
    return resolution_fcn(inds,axis=axis)
        