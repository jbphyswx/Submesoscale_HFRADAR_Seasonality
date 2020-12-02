# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:23:34 2020

@author: Jordan
"""
import numpy as np
import xarray as xr
import dask
import dask.array as da
import itertools
import sys
import inspect
import metpy.calc as mpcalc
import itertools
import functools 


#test yes

import personal.math

def is_numpy_type(x):
    """
    Somewhat unpythonic way of checking if an array is a numpy type
    """
    return type(x).__module__ == np.__name__


def isempty(x):
    
    try:
        return x.size == 0
    except:
        pass
    
    try:
        return len(x) == 0
    except:
        raise ValueError("Could not evaluate emptyness from given type " + type(x).__name__)
        
    
def isvector(x):
    """ Tries to replicate what matlab has where dimensions are eitehr (n,) or (n,1) (or transpose) """
    nd = np.ndim(x) # store so no recompute
    if nd == 0:
        return False # is scalar
    elif nd == 1:
        return True
    elif nd ==2:
        if 1 in np.shape(x):
            return True
        else:
            return True # is Matrix (2D)
    else:
        return False  # is ND array
    
def is1dvector(x):
    """ Tells you if your array is 1d """
    return np.ndim(x) ==1 # store so no recompute

    
def first_non_singleton_dim(x):
    if np.isscalar(x):
        return 0
    sz = np.shape(x) # is a tupl
    dim = next((x for x, val in enumerate(sz) if val > 1), 1)    #dim = np.argmax(sz > 1) # argmax will return the first answer, 0 if no asnwer, but only works if not a tuple
    #return max(dim,1)
    return dim
    

def replace(a, ind, v, mode='raise'):
    # np.put, but not in place
    a2 = np.copy(a)
    a2.put(ind,v,mode)
    return(a2)
    
    
def slice_select(x,dim,ind,return_indices=False):
    """ selects slice ind in dimension dim, if you have specific selections or a range use range or list/tuple?
    if you want matlab style slicing that maintainst dimensions, put the index(es) you want in a list"""
    
    indices = (slice(None),) * dim + ((ind),)
    return x[indices] if not return_indices else indices # return indices if requested
    
def any_select(x,dims_inds,return_indices = False):
    """ selects slices, dims_inds is a dictionary mapping inds to the indices you want to cut in that dimension
        if you want matlab style slicing that maintains dimensions, put the index(es) you want in a list"""
    nd = np.ndim(x)
    if not set(dims_inds.keys()).issubset(set([i for i in range(nd)])):
        raise ValueError('Inavlid key passed, dimension keys must be valid integer dimensions of x')

    indices = tuple(dims_inds.get(i,slice(None)) for i in range(nd)) # selects the inds if theyre in the dict, else just slices all
    return x[indices] if not return_indices else indices

def colon_any_select(x,dims_inds,return_indices = False):
    """ selects slices, dims_inds is a dictionary mapping inds to the indices you want to cut in that dimension
        if you want matlab style slicing that maintains dimensions, put the index(es) you want in a list
        Equivalent of slicing with a colon, so dim_inds must be of form {dim:[start_ind, stop_ind]}"""
    nd = np.ndim(x)
    if not set(dims_inds.keys()).issubset(set([i for i in range(nd)])):
        raise ValueError('Inavlid key passed, dimension keys must be valid integer dimensions of x')
        
    indices = [None]*nd
    for dim in range(nd):
        rng = dims_inds.get(dim,None)
        if rng is None:
            indices[dim] = slice(None)
        else:
            indices[dim] = slice(dims_inds[dim][0],dims_inds[dim][1]) # start stop slice
    indices = tuple(indices)
    # selects the inds if theyre in the dict, else just slices all
    return x[indices] if not return_indices else indices

def is_equal_along_dimension(x,dim):
    """
    Checks that all the slices along the dimension dim are equal, that is the array x is constant along dimesion d
    """
    return np.all([np.array_equal(slice_select(x,dim,i), slice_select(x,dim,i-1)) for i in range(x.shape[dim])])


def pad(x,dim=-1,num_l=1,num_r=1,**kwargs):
    """
    Takes array x and pads along dimension dim by repeating the n end slices before the beginning and the n begining slices after the end.
    Useful for differencing with np.diff
    """
    nd = np.ndim(x)
    if dim == -1: # make dim non negative
        dim = nd-1
    pad_vals = [(num_l*(i==dim),num_r*(i==dim)) for i in range(nd) ] # pad only on dimension dim
    return np.pad(x,pad_vals,**kwargs)
    
    
def align_along_dimension(v,dim):
    """
    Takes a 1D or 2D vector and aligns it along dimension dim as a dim-d vector, d>=1 (at least 2 dimensions)
    """
    out_size = [1] *np.max([2,dim+1])
    out_size[dim] = len(v)
    return np.reshape(v.flatten(),out_size)

def atleast_nd(arr,n):
    """
    Adds trailing singleton dimensions as necessary to ensure array arr is at least n dimensional
    """
    return np.expand_dims(arr, tuple(range(arr.ndim,n)) )
    
def apply_along_axis(f,axis, arrays, *kargs,  protected_dims=[], f_kwargs={}): # this can probably replace apply_along axis -- just assume no protecteddims as default
    """
    Applies arbitrary function f to slices along dimensions from arrays in arrays (assumes they're all the same size)
    Assumes f either preserves the dimension or collapses it to a 0-D scalar
    arrays is some set of arrays (like a list)   
    
    protected dims are dims that dont get flattened. so instead of applying to a 1D slice along dimensions, it would be an ND slice where N contains the protected dimensions. This is useful if you can vectorize part, but not all of an operation...
    e.g. taking the variance along time, but not across elevation, while lat,lon don't matter (protected dimensions). If lat and lon are large dimensions, this could lead to slower code if not specified as protected since you lose vectorization
    """

    sz   = np.shape(arrays[0])
    dims_bookeeping = np.arange(len(sz)) # for bookeeping, start in order
    def swap(iterable, swapinds):
        iterable[swapinds[0]], iterable[swapinds[1]] = iterable[swapinds[1]], iterable[swapinds[0]]
        return iterable
    # calculate what prescripted swapping will do to axes
    dims_bookeeping = swap(dims_bookeeping,[axis,-1])
    i = -2
    for pdim in protected_dims:
        dims_bookeeping = swap(dims_bookeeping,[pdim, i])
        i-=1

    
    def f_swap(a,swapaxes):
        return a.swapaxes(swapaxes[0],swapaxes[1]) #.reshape(-1, a.swapaxes(-1,axis).shape[-1])
    
    def f_swap_all(a,axis=axis,protected_dims=protected_dims):
        a = f_swap(a,[axis,-1]) # swap action axis to last
        i = -2                                                 # set to one less (swap next axis to -2)
        for pdim in protected_dims:
            a  = f_swap(a,[pdim,i]) # swap protected dims 1 earler each time and keep bookeeping 
            i -= 1
        return a
            
        
    def f_reshape(a,axis=axis,protected_dims=protected_dims):
        a = f_swap_all(a,axis=axis,protected_dims=protected_dims)
        # reshape by flattening all but the protected dims
        sh = np.shape(a) # shape after swaps
        i = -len(protected_dims)-1 # number of axes operated on
        if np.abs(i) == len(sh): # i.e. if i-1 = len(sh), we've operated on all axes... we shouldn't flatten anything... otherwise reshape below will prepend an axis at the beginning, which may not be intended. Encourage the user to do so. same as len(protected_dims)+1
            raise ValueError('All axes are either protected (cannot be flattened) or are the function axis... consider prepending at least one axis to your array that is not protected')
        return a.reshape((-1,) + a.shape[i:]) # flatten all but the protected dems and action axis which is now at the end
    
        
    sz_swap = [sz[i] for i in dims_bookeeping] # the swaps
    sz_reshape = [np.prod(sz_swap[:(i+1)])] +  sz_swap[(i+1):]
    sz_swap = tuple(sz_swap)
    
    arrays = zip(*[f_reshape(a,axis) for a in arrays]) # zip together all the aligned arrays (these are 1D arrays, idk if i should align them along dimension)
    arrays = list(arrays)
    out = [f(*a_set,*kargs,**f_kwargs) for a_set in arrays] # apply the function to each of our aligned groupings,
    sz_f = out[0].shape # the size the fcn spits out, has shape that came out of f_reshape lambda fcn minus any reduction from the fcn (e.g. np.dot reduces the dotted dimension)
    out = np.stack(out) if len(sz_f) >0 else np.stack([out]) # stack the output arrays back along the first dimension (if the fcn reduced the inputs to a scalar add it back to stack
    out = out.reshape(sz_swap[0:-1]+(-1,))  # restore original shape for other dimensions with the operated on dimension free in case its size changed
    # undo prior swaps
#     out = out.swapaxes(-1,axis) # undo the swap
    out = np.moveaxis(out,np.arange(out.ndim),dims_bookeeping) # shift dims back to where they came from
    return out    
    
    
# ------------------------------------------------------------------------------#    
    
def _apply_along_axis_no_protected_dim(f,axis, arrays, *kargs, f_kwargs={}): # old version, ok to delete at some point
    """
    Applies arbitrary function f to 1D slices along dimensions from arrays in arrays (assumes they're all the same size)
    Assumes f either preserves the 1D dimension or collapses it to a 0-D scalar
    arrays is some set of arrays (like a list)   
    """
    

    sz  = np.shape(arrays[0])

    f_reshape = lambda a,axis : a.swapaxes(-1,axis).reshape(-1, a.swapaxes(-1,axis).shape[-1]) # move desired dimension to last and flatten all but last dimension
    sz_swap = list(sz)
    sz_swap[-1], sz_swap[axis] = sz_swap[axis], sz_swap[-1] # the size coming out of f_reshape
    sz_swap = tuple(sz_swap)
    
    arrays = zip(*[f_reshape(a,axis) for a in arrays]) # zip together all the aligned arrays (these are 1D arrays, idk if i should align them along dimension)
    
    out = [f(*a_set,*kargs,**f_kwargs) for a_set in arrays] # apply the function to each of our aligned groupings,
    sz_f = out[0].shape # the size the fcn spits out, has shape that came out of f_reshape lambda fcn minus any reduction from the fcn (e.g. np.dot reduces the dotted dimension)
    out = np.stack(out) if len(sz_f) == 1 else np.stack([out]) # stack the 1D arrays back along the ending dimension (if the fcn reduced the dimension readd it)
    out = out.reshape(sz_swap[0:-1]+(-1,))  # restore original shape for other dimensions with the operated on dimension free in case its size changed
    out = out.swapaxes(-1,axis) # undo the swap
    return out    
    
def isbetween(x,bounds,inclusive=False):
    """
    Tells you if x is between bounds = [bounds[0],bounds[1]] bounds (so splitting on first dimension)
    Inclusive = True means matching the bounds is a match
    """
    if inclusive:
        return np.logical_and(np.less_equal(bounds[0],x), np.less_equal(x,bounds[1]))
    else:
        return np.logical_and(np.less(bounds[0],x), np.less(x,bounds[1]))
    
    
        
def xr_split_by_strftime(ds,fmt_string='%m-%d-%Y',return_datestrings=False, return_data=True):
    """ 
    Returns list of tuples including time and datasets/arrays split based on formatting of time via fmt_string 
    """
    split_data = list(ds.groupby(ds.time.dt.strftime(fmt_string))) # returns list of tuples w/ (time, data)
    if return_data:
        if return_datestrings:
            return split_data
        else:
            return [x[1] for x in split_data]
    else:
        if return_datestrings:
            return [x[0] for x in split_data]
        else:
             None

def xr_split_by_linspace(ds, split_dims_and_spacings, return_slices=False, return_data=True):
    """
    A dictionary of dims and the slice lengths we need.

    """ 
    def split_by_linspace(dataset,split_dims_and_spacings, return_slices=return_slices,return_data=return_data):
        slice_indices = {}
        split_slices = {}

        for dim, spacing in split_dims_and_spacings.items():
            L = len(ds[dim])
            slice_indices[dim] = list(range(0, L, spacing))
            if slice_indices[dim][-1] < L:
                slice_indices[dim].append(L)
                split_slices[dim] = [slice(slice_indices[dim][i],slice_indices[dim][i+1]) for i in range(len(slice_indices[dim])-1)]
        for slices in itertools.product(*split_slices.values()):
            selection = dict(zip(split_slices.keys(), slices))
            if return_data:           # do return data
                if return_slices:       # do return slices
                    yield (selection,dataset[selection])
                else:
                    yield dataset[selection]
            else:                       # don't return data
                if return_slices:
                    yield selection
                else:
                    yield
    return  list(split_by_linspace(ds,split_dims_and_spacings))
            
            
def xr_split_by_chunk(ds, split_dims='all', return_slices=False,return_chunks=True):
    """
    adapted from https://ncar.github.io/xdev/posts/writing-multiple-netcdf-files-in-parallel-with-xarray-and-dask/
    """
    
    if split_dims == 'all':
        split_dims = list(ds.dims)
        
    def split_by_chunks(dataset,split_dims=split_dims, return_slices=return_slices):
        chunk_slices = {}
        for dim, chunks in dataset.chunks.items():
            if dim in split_dims: #only calculate slices for dims we wish to slice?
                slices = []
                start = 0
                for chunk in chunks:
                    if start >= dataset.sizes[dim]:
                        break
                    stop = start + chunk
                    slices.append(slice(start, stop))
                    start = stop
                chunk_slices[dim] = slices
        for slices in itertools.product(*chunk_slices.values()):
            selection = dict(zip(chunk_slices.keys(), slices))
            if return_chunks:           # do return chunks        
                if return_slices:       # do return slices
                    yield (selection,dataset[selection])
                else:
                    yield dataset[selection]
            else:                       # don't return chunks
                if return_slices:
                    yield selection
                else:
                    yield
            
    return list(split_by_chunks(ds,split_dims=split_dims))
        
    
def xr_split_by_grouping(ds,dim='time',grouping_method='sel', grouper=None):
    """
    permit grouping by sel, isel, groupby
    for sel, isel does ds.sel,isel(dim=slice())
    
    for grouby does gg = [x for x in data_mp.groupby(groupby method data_mp.time.dt.strftime('%m-%d-%Y'))]
    
    CONSIDER EXPANDING TO MULTIPLE DIMS TO CREATE A nd GRID OF OUTPUT DATASETS?
    """
    
    if grouping_method == 'sel':
        return [ds.sel(**{dim:slice(group,group)}) for group in grouper]
    elif grouping_method == 'isel':
        return [ds.isel(**{dim:(grouper[i],grouper[i+1])}) for i in range(len(grouper))]
    elif grouping_method == 'groupby':
        return [x for x in ds.groupby(grouper)]
    elif grouping_method == 'chunk': # splits by chunk along dimension dim
        pass
    else:
        raise ValueError('Inavlid grouping method.')

    return

"""
EVENTUALLY, the goal is to get things to work so that you could split a dataset by chunk or indices or strfmt etc and save...
Or any above method --- you could use anyting above this comment
Autogenerating the file names is probably the most annoying part...

An idea for this filename parsing is based on the functions written above, one should decide about speical characters:
gg = pds.xr_split_by_linspace(data_mp,{'time':12,'lon':50},return_data=False,return_slices=True)
first date of first slice is np.datetime_as_string(data_mp['time'][gg[0]['time'].stop])

it could be prudent to decide how to indicate coordinates for inclusing in filename, 

"""
    
def xr_save_by_date_strtfmt(ds, filepath, filename_prefix, fmt_string='%Y-%m-%d', **kwargs):
    split_data = xr_split_by_strftime(ds, fmt_string=fmt_string, return_datestrings=True, return_data=True) # keep both
    filenames  = [filepath + '/' + filename_prefix + '_' + x[0] + '.nc' for x in split_data]
    datasets   = [x[1] for x in split_data] 
    
    xr.save_mfdataset(datasets,filenames)
    return datasets,filenames
    
    
    
    
def xr_save_by_grouping_along_dimension(ds,filepath, filename_prefix, grouping='chunk',parallel=True,dim='time'):
    """
    Default is to save each chunk to it's own file, name autogenerated by dim
    You can also provide your own slicing for grouping for use with ds.sel(dim=slice(...))
    """
    
    return
    
def array_size_estimator(shape,fmt='MB',np_or_sys='np'):
    """
    Estimates array size in memory based on array-like input of dimensions
    """
    A = np.empty(shape)
    
    if np_or_sys == 'np':
        sz_bytes = A.nbytes
    elif np_or_sys == 'sys':
        import sys
        sz_bytes = sys.getsizeof(A)
    else:
        raiseError('Invalid Method')
    
    sizes = ['B','KB','MB','GB','TB','PB']
    spacing = 2**10 # x1024 factor between each level
    return sz_bytes / (spacing**sizes.index(fmt.upper()))
       
    
def xr_repeat(ds,n,dim,extrapolate_dim=False):
    """
    Allows you to repeat an xarray n times along dim
    You have the option to extrapolate dim in the output (e.g. continuing a time axis into the future) rather than repeating
    """
    
    ds_out = xr.concat([ds]*n,dim=dim)
    
    if extrapolate_dim:
        x = ds[dim]
        L = len(x)
        x_new = np.tile(x.values, n) # repeat to preallocate
        x_diff = x - x[0]
        for i in range(1,n):
            x_new[i*L:(i+1)*L] =  x + x_diff*i #### THIS IS NOT RIGHT!!! EITHER ATTEMPT AN EXTRAPOLATION THAT YOU THINK WORKS FOR TIME DATA OR CONSIDER ASSUMING EQUAL SPACING TO CONTINUE EXTRAPOLATION? UNCLEAR HOW ELSE TO CONTINUE FROM END BACK TO BEGINNING OF ARRAY...
        ds_out[dim] = x_new
    return ds_out
        
    
    
def contiguous_regions(boolean_array):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "boolean_array"
    d = np.diff(boolean_array)
    idx, = d.nonzero() 

    # We need to start things after the change in "boolean_array". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if boolean_array[0]:
        # If the start of boolean_array is True prepend a 0
        idx = np.r_[0, idx]

    if boolean_array[-1]:
        # If the end of boolean_array is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx

def shift_vec_to_range(v, old_range_lower_bound=None, new_range_lower_bound=0,v_range=None,inclusive_upper_bound=True):
    """
    shifts dim to be within the new range...
    v_range is the span of the values from the lower_bounds... default is the range of v...
    ... we give freedom to specify a range that is a superset of range(v) (otherwise raises an error)
    shifting is done by this distance d as necessary until in range
    
    inclusive_upper_bound defines whether or not the upper bound of the range is a valid value... if not, we do a quick fix at the end
    - useful if say you have longitude data 0...360 where the upper bound(360) is redundant
    - may be expensive to do a 2nd pass over all the data to check for equality right after doing comparisons but this fcn shouldn't be used on superhuge data anyway
    - this is also a necessary step since 
    
    could do iteratively (shift then check, up until in range then final adjustments) but we can do better wit some math...
    
    """
    # Deal w/ v_range default if not specified
    if v_range is None:
        v_range =  np.ptp(v) # set to range of input vector (basically will become a translation
    if old_range_lower_bound is None:
        old_range_lower_bound =  np.min(v) # set to range of input vector (basically will become a translation
        
    # overload greater and less than functions (gf and lf) to deal with upper bound inclusivity
    if inclusive_upper_bound:
        gf = np.greater # must actually violate bound
        lf = np.less
    else:
        gf = np.greater_equal # matching bound is a violation
        lf = np.less_equal
        
    # Ensure values of v conform to prescribed range
    if ((np.min(v) < old_range_lower_bound) or (gf(np.max(v) , (old_range_lower_bound + v_range)))):
        raise ValueError ('Values of v violate range prescribed by lower bound ' + str(old_range_lower_bound) + ' and ' + 'inclusive'*inclusive_upper_bound + ' width ' + str(v_range) + ' (saw range of ' + str([np.min(v), np.max(v)]) + ')')
        
        
    shift_direction = np.sign(new_range_lower_bound-old_range_lower_bound) # 1 (+, shift up), (0, none/return), (-, shift down)
    
    if   shift_direction ==  0: # shift none
        return v
    
    elif shift_direction ==  1: # shift up
        # if shifting up, need (lower_bound + range) + n*range to end up in range (first n such that we land >= new_lower_bound)
        n_shift = np.ceil((new_range_lower_bound -  (old_range_lower_bound + v_range)) / v_range)
        v = v + n_shift*v_range # apply 
        v[v < new_range_lower_bound] += v_range # should only need to clean up once now thanks to our math! (always use normal less than bc you dont wanna add +range to an exact match lower bound with a noninclusive upper bound). Matching lower bound is fine
        
    elif shift_direction == -1: # shift down
        # if shifting down, need lower_bound to end up in range (first n such that we land <= (new_lower_bound + range)
        # this is actually symmetric (as shifting is) so we just have to switch variables... but then we go the other direction so neg
        n_shift = np.ceil((old_range_lower_bound -  (new_range_lower_bound + v_range)) / v_range)
        v = v - n_shift*v_range
        v[gf(v,(new_range_lower_bound+v_range))] -= v_range # here we wanna use the over loaded class since matching the uppper bound is problematic if not inclusive
        
    return v



def xr_shift_coord(ds,coord='lon',old_range_lower_bound=0, new_range_lower_bound=-180,v_range=360,inclusive_upper_bound=False, shift_sorted_dim=True,copy=True):
    """
    shifts coord vector values to be within a new range using shift_vec_to_range...
    an additional functionality is that if the dim is sorted (i.e. composed of <=2 monotonic chunks after sorting) it can be circle shifted     so the resulting vector is monotonic, sorted. 
    
    Default is to shift lon from [0,360) to [-180,180) and shift to monotonic lon
    """
    if copy:
        ds = ds.copy(deep=True)
    
    # handle coord transformation
    ds[coord] = shift_vec_to_range(ds[coord].values, old_range_lower_bound=old_range_lower_bound, new_range_lower_bound=new_range_lower_bound,v_range=v_range,inclusive_upper_bound=inclusive_upper_bound)
    
    # shift to monotonic
    if shift_sorted_dim:
        ind = np.argmin(ds[coord].values) # get index of minimum
        ds  = ds.roll(shifts={coord:-ind},roll_coords=True) #shift backwards
    
    return ds

    
    
def replace_nan_from_array(A,B,copy=True,deepcopy=False):
    """
    Replace NaNs in A w/ values from B located in the same spot... A,B should be the same shape
    """
       
    if deepcopy:
        A = np.deepcopy(A)
    elif copy: # deepcopy overrides copy
        A = np.copy(A)
             
    nan_locs = np.isnan(A)
    A[nan_locs] = B[nan_locs] 
    
    return A


def get_indices(A, values, otype_constructor=list):
    """
    Return a dict mapping {val:[indices]} locating where values are loacated in array A...
    data out type created by otype_constructor
    """
    # fastest way is just to iterate over the array once... (rather than a separate list comprehension for each in )
    # not sure how to use append say with np.ndarray so start as list and then cast later
    
    # values = set(values) don't do this, incase we want duplicate values?
    out = {val:[] for val in values}
    
    for i,j in enumerate(A): # works even on numpy!
        if j in values:
            out[j].append(i)

    out.update({k: otype_constructor(v) for k,v in out.items() })
    return out
    


def reorder(x, reorder_inds,otype_constructor=list):
    """
    reorder an arraylike item to be in the order given
    """
    return otype_constructor([x[i] for i in reorder_inds]) # wrap the generator in list so it can't return a generator object
    
    

def tuple_insert(tup,val,ind, perform_checks=False):
    """
    Insert value at position in tuple
    If perform checks is off, note that out of bounds index requests will just get inserted at the end by code form used,
    failing without warning of this may not be desired behavior. 
    
    list.insert() in python does not perform this check, so you may desire to leave it off for equal functionality...
    """
    if perform_checks:
        L = len(tup)
        if ind>L:
            raise ValueError('Index out of bound for tuple of length ' + str(L))
        else:
            pass
                             
    return tup[ :ind] + (val ,) + tup[ind: ]

def tuple_replace(tup,val,ind):
    """ list conversion is faster for long tuples so we use this method
        https://stackoverflow.com/questions/11458239/python-changing-value-in-a-tuple/20161067#20161067"""
    
    out = list(tup)
    out[ind] = val
    return  tuple(out)
    
    
def dask_expand_dims(da,axis):
    """
    dask implementation of expand_dims...
    Becuase you have to specify the new chunksize, we will have to do some math to make sure we know the chunk sizes
     - this is bc a single trailing chunk of unequal size may not have the same size as the others to specify...
     - Note d.chunks will also give you 
    
    
    axis = dimension in the expanded axes where the new axis (or axes) is placed. int or tuple, same as np.expand_dims
    Note: np.expand_dims indexing uses positioning in the output, not the input axis.
    - That means axis = -1 will insert a new axis at the end of the output axis.
    - This is at odds with how list.insert() and thus tuple.insert() would work so we mod out negative numbers...

Changed in version 1.18.0: A tuple of axes is now supported. Out of range axes as described above are now forbidden and raise an AxisError.
    
    """
     
    
    if isinstance(axis,int):
        axis = (axis,) # make into array
    
    ndim= da.ndim    
    new_axis = axis # the inserted axes positions for the new_axis arg in map_blocks
    for i,ax in enumerate(axis):
        if ax < 0: new_axis = tuple_replace(new_axis,ax%(ndim+1),i) # use ax < 0 check to update tuple as few times as possible
     
    return da.map_blocks(lambda x: np.expand_dims(x,axis), new_axis=new_axis)
    
    
def apply_in_chunks(fcn,
                    chunked_arr,
                    *args,
                    output_shape=None,
                    input_chunk_sizes={0:1},
                    output_chunk_sizes={},
                    parallelize = True,
                    parallel_args={'n_jobs':4},
                    progress_bar=False,
                    verbose=False,
                    joblib_directory='./joblib_memmap',
                    mmap_output=True,
                    mmap_input=True,
                    fcn_kwargs={}):
    """
    Apply fcn in chunks along chunked_arr 
     - fcn should retain singleton dimensions in its output so chunks can be properly stacked

    supplying output_shape is mandatory because we cannot know the sizes of all the chunks (allows parallelization too)
    chunks as a dict mapping dimension to np array of chunk sizes (if int, expands to full indices)
    If unspecified assumes the entire thing is a chunk

    input and output chunk sizes should lead to the same number of chunks total,
    and the same number of chunks along each dimension retained in the output
    (i.e. you can reduce a dimension that began with only one chunk but
    otherwise the output chunk should retain the singleton dimension)

    The necessary condition is that:
        > the # of chunks implied by input_chunk_sizes/chunked_array shape and output_chunk_sizes/output_shape must match
        > the orientation of these chunks (i.e. the shapes of the chunked arrays, i.e shape order excluding singletons) must match 

     """
    
    def print_verbose(x,verbose=verbose):
        if verbose: print(x)
        return
    
    input_shape  = chunked_arr.shape
    input_size   = chunked_arr.size # same as np.prod(input_shape)
    input_ndim   = chunked_arr.ndim
    
    if output_shape is None: output_shape = input_shape
    output_size  = np.prod(output_shape)
    output_ndim  = len(output_shape) # note this could be both smaller or larger than input_ndim!!


    nchunks_in  = 1 # v these must match
    nchunks_out = 1 # ^ these must match
    input_chunk_sh  = [None]*input_ndim # shape of array indicating chunk
    output_chunk_sh = [None]*output_ndim 

    # process inputs
    for dim in range(input_ndim): # iterate over these separately because we note th
        in_ch_size = input_chunk_sizes.get(dim,input_shape[dim]) # get this dimensions chunk size, default to a single chunk
        output_chunk_sizes.setdefault(dim,in_ch_size) # default to keeping the same value as the input, but don't overwrite existing
        if isinstance(in_ch_size,int):
            factors = divmod(input_shape[dim], in_ch_size)
            if factors[1] == 0:
                input_chunk_sizes[dim] = np.array([in_ch_size]*factors[0]) # expand to chunk sizes list
            else:
                input_chunk_sizes[dim] = np.array([in_ch_size]*factors[0] + [factors[1]]) # expand to chunk sizes list
        input_chunk_sh[dim] = len(input_chunk_sizes[dim])
        sums = np.cumsum(input_chunk_sizes[dim])
        input_chunk_sizes[dim] = np.insert(sums,0,0)
    nchunks_in = np.prod(input_chunk_sh)

#     print_verbose(input_chunk_sizes)
    
    # process outputs
    for dim in range(output_ndim):
        if dim in input_chunk_sizes:
            out_ch_size = output_chunk_sizes.setdefault(dim,input_chunk_sizes[dim][1:]) # default to input, but for example if input_ndim > output_ndim, out_ch_size might be empty and then we use else clause to get output_shape value
        else:
            out_ch_size = output_shape[dim]
        
        if isinstance(out_ch_size,int):
            factors = divmod(output_shape[dim], out_ch_size)
            if factors[1] == 0:
                output_chunk_sizes[dim] = np.array([out_ch_size]*factors[0]) # expand to chunk sizes list
            else:
                output_chunk_sizes[dim] = np.array([out_ch_size]*factors[0] + [factors[1]]) # expand to chunk sizes list

        output_chunk_sh[dim] = len(output_chunk_sizes[dim])
        sums = np.cumsum(output_chunk_sizes[dim])
        output_chunk_sizes[dim] = np.insert(sums,0,0)  
    nchunks_out = np.prod(output_chunk_sh)
   
    # perform checks (all dims gotta match up before the last non 1 value)
    if nchunks_in != nchunks_out:
        raise ValueError('given chunk sizes generate mismatched chunks:' + \
                         'input chunk shape '  + str( input_chunk_sh) + ' (' + str(nchunks_in) + ' input chunks), ' + \
                         'output chunk shape ' + str(output_chunk_sh) + ' (' +str(nchunks_out) + ' output chunks)')
    if not tuple(filter(lambda x: x!=1,input_chunk_sh)) == tuple(filter(lambda x: x!=1,input_chunk_sh)):
        raise ValueError('Cannot coerce chunk order to equivalence,'  \
                        'input chunk shape '  + str( input_chunk_sh) + ' (' + str(nchunks_in) + ' input chunks), ' + \
                         'output chunk shape ' + str(output_chunk_sh) + ' (' +str(nchunks_out) + ' output chunks)')
                        

    out = np.full(output_shape,np.nan) # preallocate # we wil have to do some math with input and output chunk sizes


    # chunks will be numbered 1 to #chunks, ordered in lexicographic order along dimensions...
    # mapping chunk number to the actual chunk 
    def get_chunk_position(chunk_num,chunk_sh): return np.stack(np.unravel_index(chunk_num,chunk_sh)).T

    def process_chunk_num(chunk_num,out=out,chunked_arr=chunked_arr):
        ''' helper to allow joblib to be a generator '''
        in_pos      = get_chunk_position(chunk_num,input_chunk_sh) # returns a tuple
        in_inds     = {dim: input_chunk_sizes[dim][in_pos[dim]:(in_pos[dim]+2)] for dim in range(input_ndim)}
        in_selected = colon_any_select(chunked_arr,in_inds,return_indices=False) # return_indices=False means just give me the array

        out_pos      = get_chunk_position(chunk_num,output_chunk_sh) # returns a tuple
        out_inds     = {dim: output_chunk_sizes[dim][out_pos[dim]:(out_pos[dim]+2)] for dim in range(output_ndim)}
        out_select   = colon_any_select(out,out_inds,return_indices=True) # return_indices=True means give me the indices not the array
        out[out_select] = fcn(in_selected,*args, **fcn_kwargs)
        return chunk_num

    # parallelize this loop
    it = range(nchunks_in)   
    if progress_bar:
#         from tqdm import tqdm
        from tqdm.auto import tqdm
        it = tqdm(it)
    if parallelize:
        print_verbose('running in parallel with ' + str(nchunks_in) + ' chunks, over ' + str(parallel_args['n_jobs']) + ' jobs')
        print_verbose('Parallel arguments: ' +str(parallel_args))
        from joblib import Parallel, delayed, dump, load
        import os
        import shutil
        # try using memory mapping (probably on chunked_arr, and out since they're biggest----------------------
        if mmap_input or mmap_output:
            folder = joblib_directory
            try:
                print_verbose('handling folder...')
                os.mkdir(folder)
            except FileExistsError:
                print_verbose('...remaking')
                shutil.rmtree(folder)
                os.mkdir(folder)

        if mmap_input:
            data_filename_memmap = os.path.join(folder, 'data_memmap')
            print_verbose('...mapping to: ' + data_filename_memmap)
            dump(chunked_arr, data_filename_memmap)
            chunked_arr = load(data_filename_memmap, mmap_mode='r')
        if mmap_output:
            output_filename_memmap =  os.path.join(folder, 'output_memmap')
            out = np.memmap(output_filename_memmap, dtype=out.dtype,
                       shape=output_shape, mode='w+')
        
        #---------------------------------------------------------------------------------------------------------
        Parallel(**parallel_args, verbose=0)(delayed(process_chunk_num)(chunk_num,out=out,chunked_arr=chunked_arr) for chunk_num in it)
        #---------------------------------------------------------------------------------------------------------
        if mmap_input or mmap_output:
            try:
                print_verbose('...cleaning up')
                shutil.rmtree(folder)
            except:  # noqa
                print_verbose('Could not clean-up automatically.')

    else:
        for chunk_num in it:
            process_chunk_num(chunk_num,out=out,chunked_arr=chunked_arr)
    return out

def equal_with_nan(x1,x2,**kwargs):
    """
    uses logic to compare arrays with nans, assuming the nans are equal...
    see https://stackoverflow.com/questions/41914226/how-to-compare-two-numpy-arrays-with-some-nan-values
    - the other option is allclose option, see https://stackoverflow.com/questions/10710328/comparing-numpy-arrays-containing-nan,
      but I decided it was inferior.
    """
    
    return np.all((x1 == x2) | np.isnan(x1) | np.isnan(x2))
    
    
def count_value(arr, axis=None, id_fcn = lambda x: ~np.isnan(x), **kwargs):
    """ counts values in arr identified using id_fcn. Default is ~np.isnan(x) which locates non np.nan values"""
    return np.count_nonzero(id_fcn(arr),axis=axis,**kwargs)

   
def subset_dict(dct, subset_keys, intercept=False):
    """
    Returns a subset of dictionary dct using keys in array subset_keys...
    If intercept is True, allows subset_keys to include keys outside of dct without failing
    """
    
    if intercept:
        return {key:dct[key] for key in set(subset_keys) & set(dct.keys())} # only return if in both
    else:
        return {key:dct[key] for key in subset_keys} # return normallly, will fail on key outside dct
    
    
    
def nearest(arr, values,pre_sorted=False, sort_threshold=2):
    """
    A function to get the values and corresponding indices in array arr nearest to those in array values
    """
    n = len(values) # number of values we're looking for
    
    if not pre_sorted:
        sort_inds        = np.argsort(np.ravel(values)) # 
        values_sorted    = values[sort_inds]
    else:
        sort_inds = np.arange(n)
        
    
    idx = np.searchsorted(values_sorted, arr, side="left")    # finds index in sorted values you'd have to insert (before <-- side="left") to keep sorted order
    

    idx[idx==n] = n-1 # this is fine since in our check later, it will not be true |arr - values_sorted[n-2]| will be smaller than |arr-values_sorted[n-1]|
                     # it could be true that we violate idx > 0 though after a shift down... but then there would have only been one value anyway
    

    shift_down = np.less(np.abs(arr - values_sorted[idx-1]) , np.abs(arr - values_sorted[idx])) # if the item below is smaller tahn the one we\re inserting before, shift down
    end_or_shift_down = np.logical_or(idx == n , shift_down) # if the previous condition is true or if we have to insert at end (to the left of index n which doesnt exist), meaning we're closest to final index n-1, shift down
    shift_down_final = np.logical_and(idx>0, end_or_shift_down) # if the index is 0, do not shift down
   
        
    idx[shift_down_final] -= 1 # shift down values as necessary since search_sorted only gives the bin to put items in not the nearest
    values = values_sorted[idx]
    
    
    mapping =  dict(zip(np.arange(n), sort_inds ))
    indices = map_values(idx, mapping) # add in and handle sorting later
    
    return (values, indices)


def map_values(arr, mapping, small_mapping=False):
    """
    Map the data to in array using the mapping to the bins? idk i forgot how this works lmao
    """
    
    if small_mapping:
        arr = arr.copy()
        for value in mapping.keys():
            arr[arr==value] = mapping[value]
        return arr
            
    
    # Should be much faster for large mappings...
    pallette      = {pal:pal for pal in np.unique(arr)} # this is sorted!
#     set_pallette  = set(pallette) # set for faster membership checks later
#     mapped        = pallette.copy()
    
    map_keys      = list(mapping.keys())
    pal_keys      = set(pallette.keys())
    for key in map_keys:   # iterate through mapping
        if key in pal_keys:
            pallette[key] = mapping[key] # this is where the mapping is put into the mapping we have
        else:
            # print('key not in array')
            pass
        
    pal_keys = np.array(list(pallette.keys()))
    pal_vals = np.array(list(pallette.values()))
    index = np.digitize(arr.ravel(), pal_keys, right=True) # digitize will do the mapping and replacement
    return  pal_vals[index].reshape(arr.shape) # return and reshape

def xr_variables_only(ds):
    """
    Implements for datasets http://xarray.pydata.org/en/stable/generated/xarray.Dataset.variables.html, but removing the coordinates
    """
#     return {x:ds[x] for x in ds.variables if x not in list(ds.coords)}
    return list(KE.data_vars)

def isiterable(x):
    try:
        iter(x)
        return True # if no error, the instance is an iterable
    except TypeError:    
        return False # The instance is not an iterable")
    
def isfunc(obj): return callable(obj)


def xr_resolve_type(data, *args, otype=xr.core.dataarray.Dataset, name='var', **kwargs):
    """
    Otype can be
        xr.core.dataarray.Dataset,   'dataset'
        xr.core.dataarray.DataArray, 'dataarray'
        numpy.ndarray,               'numpy'
        dask.array.core.Array,       'dask' 
        
    If going numpy or dask array to xarray product, naming uses name, which defaults to 'var'
    For datasets, with >1 data variables, should fail to convert to other data types (for dataarray will stack on new 'variable' dimension, see http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_array.html#xarray.Dataset.to_array )
    but you could loop over the array yourself (or use dim='<variable>' argument in kwargs to change in general)
    """
    
    type_conversions = {'dataset'    : xr.core.dataarray.Dataset,
                        'dataarray'  : xr.core.dataarray.DataArray,
                        'numpy'      : np.ndarray,
                        'dask'       : dask.array.core.Array,
                        'data'       : None
                       }
    
    
    type_constructors = {#intype:                          out type,                     to out type
                          xr.core.dataarray.Dataset    : { xr.core.dataarray.Dataset   : lambda x: x,
                                                           xr.core.dataarray.DataArray : lambda x: x.to_array(*args, **kwargs),
                                                           np.ndarray                  : lambda x: np.array(x.to_array(*args, **kwargs).data),
                                                           dask.array.core.Array       : lambda x: da.from_array(x.to_array(*args, **kwargs).data),
                                                           None                        : lambda x: x.to_array(*args, **kwargs).data
                                                         },
                          xr.core.dataarray.DataArray  : { xr.core.dataarray.Dataset   : lambda x: x.to_dataset(*args, **kwargs),
                                                           xr.core.dataarray.DataArray : lambda x: x,
                                                           np.ndarray                  : lambda x: np.array(x.to_array(*args, **kwargs).data),
                                                           dask.array.core.Array       : lambda x: da.from_array(x.to_array(*args, **kwargs).data),
                                                           None                        : lambda x: x.data
                                                         },
                          np.ndarray                   : { xr.core.dataarray.Dataset   : lambda x: xr.DataArray(x, name=name,*args, **kwargs).to_dataset(),
                                                           xr.core.dataarray.DataArray : lambda x: xr.DataArray(x, name=name,*args, **kwargs),
                                                           np.ndarray                  : lambda x: x,
                                                           dask.array.core.Array       : lambda x: da.from_array(x,*args, **kwargs),
                                                           None                        : lambda x: x
                                                         },
                          dask.array.core.Array        : { xr.core.dataarray.Dataset   : lambda x: xr.DataArray(x, name=name,*args, **kwargs).to_dataset(),
                                                           xr.core.dataarray.DataArray : lambda x: xr.DataArray(x, name=name,*args, **kwargs),
                                                           np.ndarray                  : lambda x: np.array(x,*args, **kwargs),
                                                           dask.array.core.Array       : lambda x: x,
                                                           None                        : lambda x: x
                                                         }
                         }
    
    
    if isinstance(otype,str): otype = type_conversions[otype.lower().replace(" ","")] # convert otype specification strings to otypes  
    intype = type(data)

    return type_constructors[intype][otype](data)
        

def get_package(obj, return_str=False):
    mod = inspect.getmodule(obj)
    if mod is None: return None
    base, _sep, _stem = mod.__name__.partition('.')
    return sys.modules[base] if not return_str else sys.modules[base].__name__


def nanclip(data,axis=-1,representative_variable=None,return_slices=False):
    """
    returns data clipped to the smallest bounding box such that there exists no vector along dimension dim that is all nan
    i.e. clipping [time,lat,lon] data along dim 0 (time) gives the data clipped to a bounding box in lat/lon space that has no all-nan vectors along time
    
    works fine on numpy/dask data, but on xarray dataset/dataarray will maintain coords
    for dataset, uses representative_variable, if provided, or the first data_variable
    """
    
    
    

    d_type = type(data)
        
    
    if d_type is xr.core.dataarray.Dataset:
        if representative_variable is None: representative_variable = list(data.data_vars)[0]
        slices = nanclip(data[representative_variable],axis=axis,representative_variable=None,return_slices=True)
        return slices if return_slices else data.isel(slices)
    
    sh = data.shape # works on dask/numpy/dataarray
    
    if d_type is xr.core.dataarray.DataArray:
        dims = data.dims
        if isinstance(axis, int):
            coord = data.dims[axis] # get name of dimension we're working on
        elif isinstance(axis,str): 
            coord = axis
            axis  = data.get_axis_num(coord) # get the number of the dimension, same as  data.dims.index(coord) 
    else:
        dims = np.arange(data.ndim) # just the dimension numbers
    ndims = len(dims)
    inds = {dims[i]:np.arange(sh[i]) for i in range(ndims) if i is not axis} # index positions, use dim names from above
    
    coords =  np.meshgrid(*tuple([inds[dims[i]] for i in range(ndims) if i is not axis]), indexing='ij') # ij maintains order of dims and shapes, should unpack neatly from inds.keys, skip the axis dimension
    coords =  np.stack(tuple(x.flatten() for x in coords),axis=1) # flatten down to list like format with dims along dimension 0 while listing along dimension 1
    
    if d_type is xr.core.dataarray.DataArray:
        mask = ~np.all(np.isnan(data.data), axis=axis)  
    else:
        mask = ~np.all(np.isnan(data     ), axis=axis)  
    # note mask has the given axis flattened out now

    mask = mask.flatten() # should work okay w/ the axis flattened out
    coords = coords[mask]
    
    slices = {dims[i]:slice(None) for i in range(ndims)} # we will populate this
    for dim in range(ndims):
        if dim < axis:
            col = dim
        elif dim == axis:
                pass # do nothing here since this axis isn't included
        else: # dim > axis
            col = dim-1 # since we didn't include axis in coords, then we gotta shift down
            
        if dim is not axis:
            lims = personal.math.range_bounds(coords[:,col])
            slices[dims[dim]] = slice(*lims)
    
    if d_type is xr.core.dataarray.DataArray:
        return slices if return_slices else data.isel(slices)
    else:
        slices = tuple(slices[i] for i in range(ndims))
        return slices if return_slices else data[slices]

    
    
def rolling_groups(arr,axis=-1,window=1,min_count=1,center=True, periodic=False, axis_vals = None,return_axis=False,verbose=False):
    """
    use naive indexing to get around xarray's horrible horrible implementation
    
    should be slow.... in principle because we don't know how much we can actually load into memorry so we repeatedly end up loading the same data through dask's compute calls... 
    ... eventually, we should aim to implement a data streaming way of doing this and calculate 
    ... or a rolling method where we calculate the weights of the data actually loaded so far and then the update step for stepping along only has to load one more piece (would also need to track which data is falling off the back end..., even with edge cases)
    ... ^^^ IMPLEMENT THIS AT SOME POINT!!! CAN'T BE PARALLELIZED BUT SHOULD BE MUCH FASTER SINCE YOU ONLY LOAD EACH FRAME ONCE AND THEN DO WEIGHT OPERATIONS THE REST OF THE TIME (OR MAYBE NOT, TRACKING WEIGHTS REQUIRES AT LEAST THE SIZE OF THE WINDOW  IN WEIGHTS ALONG AXIS IN MEMORY)
    
    - we could add a step implementation, but you could also just step your data before...
    
    - window leans forward if is odd and centered (i.e. [_ center _ _]) for window of size 4
    
    - you can't edit dask array, so returns a list of dask objects that vary along the desired dimension
    
    - you can call your reduction fcn on the outputs yourself
    """
    
    def print_verbose(x,verbose=verbose):
        if verbose: print(x)
        return
    
    
    
    sh     = arr.shape
    ndim   = len(sh)
    if axis < 0: axis = ndim + sh # resolve axis to be positive definite
        
    L = sh[axis] # len along dimension
    out_sh = sh[:axis] + (1,) + sh[(axis+1):]

    out = [None] * L
    
    if axis_vals is None: # placeholder for data
        axis_vals = np.arange(L)
    axis_out = axis_vals.copy()
    
    
    for i in range(L): # PARALLELIZE THIS
        if center: # index i at center of window
            span    = window // 2
            is_even = 1 - (window%2)
            start   = i - span + is_even # lean forward if even
            end     = i + span + 1       # slice doesn't include final value, so add 1
            axis_out[i] = axis_vals[i]

            if periodic: 
                # this can use the negative indices where necessary to actually do the looping, but you can only have increasing indices so... gotta uhh... maybe chain two slices together if necessary using strictly positive  i.e. [start->L] + [0->end] at both beginning and end of loop
                raise ValueError('periodic not implemented yet, you could do it now lmao')
            else: # don't let go beyond array bounds
                start   = max(start, 0) 
                end     = min(end, L)
                L_slice = end - start
                if L_slice < min_count:
                    # assign nan to this value
                    out[i] = da.empty(out_sh) * np.nan
                else:
                    indices = (slice(None),) * axis + (slice(start,end),) # create slice, only have to slice up to desired axis, rest are assumed whole
                    out[i]  = arr[indices]
                        
            
        else: # do not center,
            raise ValueError('uncentered means not implemented yet, you could do it now lmao')
    
    return (out, axis_out) if return_axis else out



def assign_x_y_to_lat_lon_dataset(ds, dims_mapping, assign_coord_grids = True, assign_xy_grids=True):
    """
    assigns 2d cartesian coordinates a dataset with lat lon... 
    dims_mapping = e.g. {'t':0,'y':1,'x':2}, mapping the x,y variables to their actual number dimensions... (necessary because lat and lon could have different names)
    
    # -- to do --> implement a more thorough version of this that can map and broadcast arbitrary coordinates into arbitrary other sets based on their input dimensions and desired output dimensionality (would just use broadcasting and meshgrid, and perhaps a flag for lat/lon to x/y type conversions or just accept a conversion fcn for each desired output coord)
    """
    dx, dy = mpcalc.lat_lon_grid_deltas(ds.lon.values,ds.lat.values) # we havent the degrees east metadata to rely on :/, so gotta load values
    dx = np.expand_dims(dx,0)
    dy = np.expand_dims(dy,0)
    y  = personal.math.dq_to_q(np.array(dy),dims_mapping['y'])[0]
    x  = personal.math.dq_to_q(np.array(dx),dims_mapping['x'])[0]
    if assign_coord_grids:
        latg,long               = np.meshgrid(ds.lat,ds.lon,indexing='ij') # 'ij' to match the shape of x,y
        ds = ds.assign_coords({  "latg": (("lat","lon"),latg),   "long": (("lat","lon"),long)})
    if assign_xy_grids:
        ds =  ds.assign_coords({"x": (("lat","lon"),x), "y": (("lat","lon"),y) })
    return ds
    
    
    
def package_args(*args,**kwargs):
    """
    A shortcut for packaging up arguments into tuples (args) and dicts (kwargs) to be passed around
    Useful for example, for copying the arguments copied into a fcn to another location without having to convert names to strings and equals to colons
    """
    return {'args':args,'kwargs':kwargs}


def nested_dict_to_xarray(nested_dict, dimension_names={}, base_fcn = lambda x: (None,x), return_dimension_keys_and_values=False, allow_auto_squeeze=True ):
    """
    Converts a nested x-array to an xarray dataarray
    
    dimensions should be a dict with the dimension names for each level/axis, i.e. {0:'x',1:'y',2:'z',...}. A dict is a useful framework since you can skip names
    
    base_fcn determines what to do with the data once we reach a non dictionary...
    - It must return 2 values, an index and the data, both must be 1-D numpy arrays, or the index can be None
    -- we take the first value or index to be the dimension values, and the 2nd to be the data. The dimension values are read at the beginninng/exploratory phase and again assumed to never change.
    
    allow_auto_squeeze=True means we'll accept `D (n,1) and (1,n) shape arrays from base_fcn for index,data in addition to (n,) 1D arrays
    """
    
    
    # Exploratory Phase.
    if dimension_names is None:
        dimension_names = {}
    dimension_values = {key:None for key in dimension_names.keys()}
        
    at_base     = False
    dim_num     = 0
    explorer    = nested_dict
    while not at_base:
        if dimension_names.get(dim_num,None) is None:
            dimension_names[dim_num] = 'dim'+str(dim_num)    

        if not isinstance(explorer,dict): # rock bottom
            at_base     = True # stop here
            index, data = base_fcn(explorer)
            if index is None:
                index = np.arange(len(data))
            if allow_auto_squeeze: # accept (n,1) and (1,n) shape arrays by squeezing out extra dims, but if that dim isn't singleton will fail the shape test later on
                index, data = index.squeeze(), data.squeeze()
            if not isinstance(data,np.ndarray) and (data.ndim == 1) and (index.ndim == 1): # test to save us heartache later
                raise ValueError('argument base_fcn must return output index,data where data must be a 1D numpy array, check the types and shapes of your function output')
            dtype       = data.dtype
            dimension_values[dim_num] = index
        else:
            dimension_values[dim_num] = list(explorer.keys())
            explorer = explorer[dimension_values[dim_num][0]] #this dict is mandated to be self similar so just take the first name to continue our exploratory step
        dim_num += 1 # level up if not at the base and keep going. If we ended, we still have added one to it for the last dim that is the axis of the 1D vectors


    
        
    data_sz = tuple(len(dimension_values[lev]) for lev in range(dim_num)) # tuple for shape of data
    data    = np.full(data_sz, np.nan, dtype=dtype)
    
    
    # create a generator with all combinations of indices from data_sz, then step through it and fill the dataset
    
    coordinates = tuple(range(data_sz[i])   for i in range(dim_num-1))
    
    for coordinate in itertools.product(*coordinates): # the coordinate indices, excluding the last dim since that's filled in via our vector, itertools returns tuples so we can index data with tuple indexing directly
        out =  functools.reduce(dict.__getitem__, tuple(dimension_values[dim][ind] for dim,ind in enumerate(coordinate) ), nested_dict) 
        out = base_fcn(out)
        out = out[1]
        
        data[coordinate] = out # use a slick reduce operation to get the item from the list of indices, take the data from the 1st output
    
    if return_dimension_keys_and_values:
        return xr.DataArray(data, coords = dimension_values.values(), dims = dimension_names.values()), dimension_values.values(), dimension_names.values()
    else:
        return xr.DataArray(data, coords = dimension_values.values(), dims = dimension_names.values())
        

            
        