"""


"""


## Write a groupby fcn that uses bins

import numpy as np
import pandas as pd
import scipy.spatial
import itertools

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from fractions import Fraction


import personal.math


def _structure_function_diff_power(data,inds,order,method_fcn=None): # this can work on a single pair or multidimensionally
    # should, if inds is pairs [points, pairs, other data dims] -> [points, 1, other data dims]
    out    = np.abs(np.diff(x[inds] , axis=1))**order # Use fancy indexing, then diff along new axis, then abs and raies to power
#     out    = np.abs(np.diff(x.take(inds,axis=0) , axis=1))**order
    trailing_axes=tuple(np.arange(2,out.ndim,dtype=int)) # +1 so the last possible value is x.ndim which fits with our having added a dim
    # should then lead to [points, 1] bc we don't have keepdims and so lose trailing dims (note numpy needs ver 1.19.0 for keepdims)
    weight = personal.data_structures.count_value(out,axis=trailing_axes,id_fcn = lambda x: ~np.isnan(x))
    out    = np.nanmean(out, axis=trailing_axes)
    return  np.hstack((out,weight)) # shape is now ([points,1],[points,1]) -> hstack -> [points,2]


def _default_method_fcn(du,dl=None):
    return _magnitude_1(du)**2

# should be a fcn of (u,inds,order,method_fcn, distances)
def _structure_function_diff_power_decompositions(data,inds,method_fcn=_default_method_fcn, distances=None,lazy_load=False): # this can work on a single pair or multidimensionally
    """
    data       - shape should be analagous to [points, coords, ....] where coords could map over something like u_vec = [u,v], trailing dimensions something like time...
    inds       - indices for pairs of data points, should have shape [npairs,2]
    distances  - if not None, should be an array with shape [npairs, coords] specifying the distances correlating to the pairs of points selected... it's odd to have to pass this in along with inds but it's this way so distances can be precomputed and need not be repeatedly calculated here.
    method_fcn - f(du,dl)
    """
    out  = np.diff( data[inds], axis=1) # this is u [points, coords,...] --> u selections [pairs, 2, coords,...] --> du [npairs, 1 coords, ...]
    out  = np.squeeze(out,axis=1) # [npairs, 1 coords, ...] --> [npairs, coords, ...] # this is what we'll pass to our method_fcn
    out  = method_fcn(out, distances) # [npairs, coords, ...] --> SF [npairs, 1, ...], as desired, then proceed as normal
    
    trailing_axes=tuple(np.arange(2,out.ndim,dtype=int)) # +1 so the last possible value is x.ndim which fits with our having added a dim
    # should then lead to [points, 1] bc we don't have keepdims and so lose trailing dims (note numpy needs ver 1.19.0 for keepdims)
    weight = personal.data_structures.count_value(out,axis=trailing_axes,id_fcn = lambda x: ~np.isnan(x))
    out    = np.nanmean(out, axis=trailing_axes)
    return  np.hstack((out,weight)) # shape is now ([points,1],[points,1]) -> hstack -> [points,2]

def _default_dist_fcn(x,inds=None,metric='euclidean'):
    """ x is a list of points with shape (# pairs points, ndims), returns in condensed form  """
    return scipy.spatial.distance.pdist(x,metric='euclidean')[0]

def _default_dist_fcn_decomposed(x,inds=None,metric='euclidean'):
    """
    x is a list of points with shape ((# pairs points, ndims), returns distances between pairs of points in condensed form, with minimal intermediate footprint (~ n**2/2 vs n**2 )
    This does not calcluate normed distance, so you can directly have in `other data dims' directions, time, etc....
    """
    if inds is None:
        L = len(x)
        inds = personal.math.combinations(range(L),2,allow_repeats=False)     # should, if pairs are each pair in dices [points, pairs, other data dims] -> [points, 1, other data dims]
    return  np.diff(x[inds] , axis=1).squeeze(axis=1) # Use fancy indexing, then diff along new axis, and remove it


def dot_axis_1(x,y, keepdims=True):
    if keepdims:
        return np.expand_dims(np.einsum('ij...,ij...->i...', x, y), 1) # performs the dot product along the first axis of x,y ... the ellipses leave room for there to exist further dimensions
    else:
        return np.einsum('ij...,ij...->i...', x, y) # performs the dot product along the first axis of x,y ... the ellipses leave room for there to exist further dimensions

def _magnitude(  v,order=2,axis=1,keepdims=True): 
    return np.linalg.norm(v, ord=order, axis=axis, keepdims=keepdims)
def _magnitude_1(v               ,keepdims=True):
    return dot_axis_1(v,v                          , keepdims=keepdims)**.5 # einsum is allegedly faster than linalg.norm as per https://stackoverflow.com/questions/7741878/how-to-apply-numpy-linalg-norm-to-each-row-of-a-matrix/45006970#45006970

def _du_l(du,dl, return_scalars=False):
    """ The longitudinal (parallel) component of du, assuming du is a vector with coordinates over dimension 1 of same format as dx"""
    l_hat = personal.math.geometric_normalize(dl, axis=1, order=2) # given  dl = |dl| l_hat, we normalize dl to get l_hat (let l_hat = r_hat, l = r to undertand this in radial terms)
    l_hat = personal.data_structures.atleast_nd(l_hat,du.ndim) # allows for multiplication
    return dot_axis_1(du, l_hat) if return_scalars else dot_axis_1(du, l_hat) * l_hat  # we take (du · r_hat) r_hat  as the rotational component, hopefully broadcasting works fine...

def _du_t(du,dl, return_scalars=False):
    """ The transverse (perpendicular) component of du, assuming du is a vector with coordinates over dimension 1 of same format as dx"""
    # equivalent to rotating 90 degrees CCW or times array [[0, -1],[1,0]] but coded to ignore other trailing dimensions that could include z etc, so not matrix multiplication (kinda hard to define in more than 2D since there's no true reference for a transverse direction, you'd need 3rd, and possible more vectors) (see notability note  Nov 11 2020)
    # t_hat          = personal.math.geometric_normalization(dr, axis=1, order=2) # normalie the vector
    # t_hat[:,[0,1]] = t_hat[:,[1,0]] * personal.data_structures.atleast_nd(personal.data_structures.align_along_dimension(np.array([-1, 1]),1), t_hat.ndim) # align [1,-1] along the first dimension, and then make sure the array has as many dimensions as du_r so multiplying broadcasts if necessary.
    # return dot_axis_1(du, t_hat) * t_hat
    if return_scalars:  # We gotta do some math to get the actual unit vector we're comparing to to get a signed magnitude
        t_hat          = personal.math.geometric_normalize(dl, axis=1, order=2) # normalize the vector
        t_hat[:,[0,1]] = t_hat[:,[1,0]] * personal.data_structures.atleast_nd(personal.data_structures.align_along_dimension(np.array([-1, 1]),1), t_hat.ndim) # align [1,-1] along the first dimension, and then make sure the array has as many dimensions as du_r so multiplying broadcasts if necessary.
        return dot_axis_1(du, t_hat)
    else: # ez shortcut for what's above, should be same as dot_axis_1(du, t_hat) * t_hat
        #t_hat          = personal.math.geometric_normalization(dr, axis=1, order=2) # normalize the vector
        #t_hat[:,[0,1]] = t_hat[:,[1,0]] * personal.data_structures.atleast_nd(personal.data_structures.align_along_dimension(np.array([-1, 1]),1), t_hat.ndim) # align [1,-1] along the first dimension, and then make sure the array has as many dimensions as du_r so multiplying broadcasts if necessary.
        #return dot_axis_1(du, t_hat) * t_hat
        return du - _du_l(du, dl) 
    
def _default_dist_to_mapping_fcn(distances):
    """
    Allows you to map from distances to mapping w/o necessarily overwriting the distances you have
    defaults to taking mean over 1st dim (assumed to be coordinates), uses atleast_2d to ensure data is 2d if it isn't, leaving trailing dims ok....
    """
    if distances.ndim <= 1:
        return distances
    return _magnitude_1(distances, keepdims=False) # false so we drop this dim and can index structfun_data without issue
    

def structure_function(data                                                          ,
                       coords                                                        ,
                       random_subset_size      = None                                , # select a random subset of the data... for speed because pairing is factorial
                       dim                     = None                                ,
                       method_or_order         = _default_method_fcn                 , # assumes du of the form [points, scalar or coordinate dimensions, etc] and dx of the form [points, coordinate dimensions, etc] such that the dot product is over dimension 1, this should work regardless of the length of dimension 1 (i.e. the number of dimensions in your input, whether is <dx,dy> vectors or just a dr scalar for example..., but do choose the right method...)
                       groups                  = None                                , # specify if using known_mapping or groupby_fcn, done this way to provide clarity on empty
                       distances               = None                                ,
                       mapping                 = None                                ,
                       groupby_fcn             = None                                , # maps data at distances to groups
                       distance_fcn            = _default_dist_fcn                   , # should return a float
                       distance_to_mapping_fcn = _default_dist_to_mapping_fcn        , # this should drop any singleton dimensions created by distance_fcn so we can index structfun_data without error
                       value_fcn               = None                                , # should be a fcn of (u,inds,order, distances), may make use of a method fcn for instance
                       value_fcn_requires_dist = True                                , # If False, distances are not calculated if precomputed mapping (e.g. bins). If you supply precomputed mapping however, it'd be wise to also just provide precomputed distances
                       structfun_data          = None                                ,
                       sorted_output           = True                                ,
                       vectorize               = True                                ,
                       parallelize             = False                               ,
                       parallel_args           = {'n_jobs':4, 'require':'sharedmem'} ,
                       progress_bar            = False                               ,
                       verbose                 = False 
                      ):
    # parellelize only parallelizes looping over groups/iterable -- you can pass vectorized functions for calculating the distances/values for example that will speed up especially vectorization behind the scenes...
    # if not vectorized, the individual loop will not benefit much from vectorization since we'll go point by point which probably is pretty small (say a small time vector for each lat/lon datapoint/row... if not you could write a vectorized fcn again still and pass it in
    """
    Structure function of order n on scalar field u defined as SF_n(r) =<|u(x+r)−u(x)|^n>, for distance r
    ------------
    If dimension 0 of data/coords has len L, this function makes use of
    > INDEX LEXICOGRAPHICAL ORDER (i.e. [(0,0),(0,1),...,(0,n), (1,0),(1,1),...,(1,L), ..., (L,0),(L,1)...,(L,L)]) for organization
    ------------
    Inputs:
    
    data          - The input data (of some np.ndarray like type) with the coordinate dimensions for the the structure function flattened
                    over the 0'th dimension.
                    E.g. if we have data in x,y,z,t and want the structure function over x,y, we should flatten the x and y dimensions,
                    yielding an array with shape [s_0, s_1, s_2] = [s_x*s_y, s_z, s_t]
                    If necessary do this outside this function, possibly with flatten_dimensions from personal.data_structures
                    
    coords        - The input coordinates as an array of shape [s_0, n_coords] where s_0 is the                 
                    > This fcn uses matrix arithmetic to speed itself up by broadcasting arithmetic over trailing dimensions
                    
                    
    order         - The structure function order
    bins          - We can create distance groupings using bins, default (None) lists every found distance uniquely
    mapping - If we've already calculated the mappings between position in array to distance groups in our structure function output,
                    this can be used to save time.
                    You should provide simply the mapping as an array containing the mapping between groups and distances,
                    and a list of group indices to match the data pairs we have in index lexographical order 
                    e.g. with groups [  0.1,   0.33,   1.40,   3.19] mapping could be [0.1, 1.40, 0.1, 0.1, 3.19, 0.33,...]
                      or with groups ['1', '2', '3', '4']            mapping could be ['4', '2', '1', '4', '3',...]
                      
                    Note since the # of pairings could be quite large, one could imagine using a generator that yields these values
                    based on some underlying fcn
                    
                    
    groupby_fcn   - Default is None, but accepts fcn of groupby_fcn(dist)
                    Useful for example for grouping data into bins or other such collections
                    Must return group i.e. ([0.33,1.40]) or (1.40) or 2 '2' or whatever
                    
    groups        - Default will just use the set of all outputs from groupby_fcn 
                    
    distance_fcn  - Calculates the distances between all pairs of points in index lexicographic order
                    > You can pass any function as metric that calculates distances between pairs of points (i.e. one for lat/lon points)
                      I decided to keep this as a lambda fcn rather than hard coding pdist in case you want to pass kwargs to pdist
                      or use something completely different
    ------------
    Output has form {dist:value}, note dist could be a bin (x1,x2) or something like that, as long as hashable
    ------------
    
    Regardless of input size dimension, process is as follows:
    
    In INDEX LEXICOGRAPHICAL ORDER (i.e. [(0,0),(0,1),...,(0,n), (1,0),(1,1),...,(1,n), ..., (n,0),(n,1)...,(n,n)]) along dimension 0:
    > generate unique pairings itertools or something else? apply metric_fcn (i.e. abs(mean()) to get the sorting criteria
    > ... or provide 
    groupby either the bins or unique value? if you already have the ravel_indices you could have a shortcut grouping criteria
    
    For applying along a dimension, use apply_along_dimension or some similar tactic... (can speed up if the distance mapping is repeated)
    
    Remains to be seen how plays wit dask but should work? (pure dask not xarray)
    
    We assume if the full vectors don't fit in memory, and iterator/generator form is too slow to be tenable....
    If you wish this to work in this way, try memmap'd arrays, chunking, dask, precalculating the fcn, or a random point subset
    """
    
    # HANDLE METHOD FOR CALCULATING STRUCTURE FUNCTION TO INSERT INTO THE VALUE FCN ----------------------------------------------------------------------

    
    
    if personal.data_structures.isfunc(method_or_order):
        method_fcn = method_or_order # is already a function
        
    elif isinstance(method_or_order, int):
        def method_fcn(du,dl=None,method_or_order=method_or_order): return _magnitude_1(du)**method_or_order # magnitude of du times du·du = |du|**3

    elif isinstance(method_or_order, str):
        method_or_order = method_or_order.lower()
        if any(map(method_or_order.__contains__, ['second','2nd', '2', 'd2'])):            #  we're examining second order structure functions

            # these submethods will fail on scalar (1D inputs) since there will be no subcomponents to obtain
            if   any(map(method_or_order.__contains__, ['transverse', 'd2t', '2t'])):
                def method_fcn(du,dl): # In https://doi.org/10.1002/2016GL069405 Section 2, u_t is defined equivalently to r_hat dot (u_j , -u_i) if you go thru the cross product (i.e. t_hat is 90 degrees right of l_hat, clockwise). We could also calculate it using our calculation for du_r
                    du_t           = _du_t(du, dl)
                    return dot_axis_1(du_t,du_t)
                
            elif any(map(method_or_order.__contains__, ['longitudinal', 'd2l', '2l'])):
                def method_fcn(du,dl):
                    du_l = _du_l(du,dl) # given dl = |dl| l_hat, we normalize dl to get l_hat and take du · l_hat  as the rotational component
                    return dot_axis_1(du_l,du_l)

            elif any(map(method_or_order.__contains__, ['rotational', 'd2r', '2r'])):
                 raise NotImplementedError # these are properties of the flow field that we would have to calculate, see https://doi.org/10.1002/2016GL069405
            elif any(map(method_or_order.__contains__, ['divergent', 'd2d', '2d'])):
                 raise NotImplementedError # these are properties of the flow field that we would have to calculate, see https://doi.org/10.1002/2016GL069405
        
            else: # assume is just the default operator, D2 =  |du|**2 = du·du = du_l·du_l + du_t·du_t = du_l**2 + du_t**2 = D2l + D2t
                def method_fcn(du, dl=None): return _magnitude_1(du)**2 # = _default_method_fcn(du, dl)
        
        
        elif any(map(method_or_order.__contains__, ['third','3rd', '3', 'd3'])): #  we're examining third order structure functions
            if   any(map(method_or_order.__contains__, [    'diagonal consistent'  ])):                                    # du_l**3
                def method_fcn(du,dl): return _du_l(du,dl, return_scalars=True)**3
            elif any(map(method_or_order.__contains__, ['off-diagonal inconsistent'])):                                    # du_l**2 * du_t
                def method_fcn(du,dl): return _du_l(du,dl, return_scalars=True)**2 * _du_t(du,dl, return_scalars=True)**1
            elif any(map(method_or_order.__contains__, [    'diagonal inconsistent'])):                                    # du_l    * du_t**2
                def method_fcn(du,dl): return _du_l(du,dl, return_scalars=True)**1 * _du_t(du,dl, return_scalars=True)**2
            elif any(map(method_or_order.__contains__, ['off diagonal consistent'  ])):                                    #           du_t**3
                def method_fcn(du,dl): return                                        _du_t(du,dl, return_scalars=True)**3 # off gotta go first my g, or else it shortcicruits
            elif any(map(method_or_order.__contains__, ['off-diagonal', 'off-diag', 'non-diag', 'off diag',' non diag'])): # sum of off-diagonal terms above, equal to du_t (du·du) = du_t (du_l**2 + du_t**2) = (du_t du_l**2 +      du_t**2)
                def method_fcn(du,dl): return _du_t(du,dl, return_scalars=True) * dot_axis_1(du, du)
            elif any(map(method_or_order.__contains__, ['diagonal'    ,     'diag' ])): # sum of     diagonal terms above, equal to du_l (du·du) = du_l (du_l**2 + du_t**2) = (     du_l**3 + du_l du_t**2)
                def method_fcn(du,dl): return _du_l(du,dl, return_scalars=True) * dot_axis_1(du, du)


            else: # assume is just the default operator, D3 =  |du|**3 = |du| du·du
                def method_fcn(du,dl=None): return _magnitude_1(du)**3 # magnitude of du times du·du = |du|**3 = (diag**2 + off_diag**2)**.5
        
    else:
        raise TypeError('Unsupported method or order specified, please choose int, function, or a suppported string')
        
    # default to using the defined default above with the right method_fcn inserted  
    if value_fcn is None:
        def value_fcn(data,inds,method_fcn=method_fcn,distances=distances):
            return _structure_function_diff_power_decompositions(data,inds,method_fcn=method_fcn, distances=distances)
        
    # ---------------------------------------------------------------------------------------------------------------------------------------------------
   

    
        

    # handle length, and take a random subset if necessary
    L = len(data)
    if random_subset_size is not None: # take subset
        print('taking random data subset')
        selections = np.random.choice(L, random_subset_size, replace=False)
        L    = random_subset_size
        data = data[selections]
        coords = coords[selections]
        # since mapping, structfun_data would be defined along the pairs, we will square form and drop the rows and columns not in selections, assuming index lexicographic order!!
        if mapping is not None:
            mapping        = scipy.spatial.distance.squareform(mapping)
            mapping        = mapping[selections][:,selections]
            mapping        = scipy.spatial.distance.squareform(mapping)
        if structfun_data is not None:
            structfun_data = scipy.spatial.distance.squareform(structfun_data)
            structfun_data = structfun_data[selections][:,selections]
            structfun_data = scipy.spatial.distance.squareform(structfun_data)
    
        
    # calculate pairs
    if mapping is None:
        if vectorize:
            pairs = personal.math.combinations(range(L),2,allow_repeats=False) # same as itertools.combinations(range(L),2) but full array
        else:
            pairs = itertools.combinations(range(L),2) 
#         npairs = L*(L-1)//2 # L choose 2  

    # handle known and unknown options and set up grouping and distances
    if vectorize:
        
        if distances is None and (value_fcn_requires_dist or (mapping is None)): # if we didn't provide distances and the value_fcn or mapping requires it we'll have to calcluate it.
            distances = distance_fcn(coords)
        
        if mapping is None:
            mapping = distance_to_mapping_fcn(distances) # for pairs, a fcn to generate all pair distances in lexicographic order or some other mapping
            if grouby_fcn:
                mapping = groupby_fcn(mapping) # replace dists with mapping to whatever the groupby_fcn returns
            
        if structfun_data is None:
            structfun_data = value_fcn(data, personal.math.combinations(range(L),2,allow_repeats=False), method_fcn=method_fcn, distances = distances) # use on array of pairs of points ------------------------------------------------------------------------------------------------------------------------- apply value function!!

    else:
        
        if distances is None and (value_fcn_requires_dist or (mapping is None)): # if we didn't provide distances and the value_fcn or mapping requires it we'll have to calcluate it.
            distances =  map (lambda x: distance_fcn(np.vstack((x[0],x[1]))),itertools.combinations(coords,2) ) # mapping to coord pair list
            distance_fcn(coords)   
        
        if mapping is None:
            mapping = map(distance_to_mapping_fcn,distances) # map distances to whatever the mapping fcn returns (this is a generator!)
            if groupby_fcn:
                mapping = map(groupby_fcn,mapping)  # replace dists with mapping to whatever the groupby_fcn returns (this is a generator!)
        
        
        if structfun_data is None:
            structfun_data = map(lambda pair: value_fcn(data,np.array([pair]),method_fcn=method_fcn, distances = distances), itertools.combinations(range(L),2)) # ------------------------------------------------------------------------------------------------------------------------- apply value function!!
        
   
    if groups is None: # default to span of groupby_fcn output
        if vectorize:
            groups = np.unique(mapping)
            struct_fun_info = {key:[0,0] for key in groups}
        else:
            struct_fun_info = dict() # start empty so we can use our generators to fill
    else:
        struct_fun_info = {key:[0,0] for key in groups} # from keys would fucc up 
          
    
    if vectorize:
        it = groups # iterate over the distsance_bin groupings we calculated
        tqdm_total = len(groups)
    else:
        it = zip(mapping,structfun_data) # iterate over each result (note this uses a different parallel_fcn later) structfun_data is an array w/ rows [value,weight] where value might be nan for weight 0
        tqdm_total = L*(L-1)//2
        
    if progress_bar:
#         from tqdm import tqdm
        from tqdm.auto import tqdm
        it = tqdm(it,total=tqdm_total)

    if parallelize:      
        print('parallelizing')
        parallel_args.update({'require':'sharedmem'}) #ensure shared memory for dict
        from joblib import Parallel, delayed

    # Put it all together
    if vectorize: # we'll already know the groups...
        if parallelize:
            # write fcn to take dst_group,mapping=mapping,structfun_data=structfun_data and update struct_fun_info
            def parallel_fcn(dst_grp, struct_fun_info = struct_fun_info, structfun_data=structfun_data,mapping=mapping):
                dg =  structfun_data[mapping==dst_grp]
                struct_fun_info[dst_grp] = np.nansum(np.prod(dg,axis=1)) / np.sum(dg[:,1])
                return # return nothing
            Parallel(**parallel_args)(delayed(parallel_fcn)(dst_grp,struct_fun_info = struct_fun_info, structfun_data=structfun_data,mapping=mapping) for dst_grp in it)
        else: # no parallelize
            for dst_grp in it:
                dg =  structfun_data[mapping==dst_grp]
                struct_fun_info[dst_grp] = np.nansum(np.prod(dg,axis=1)) / np.sum(dg[:,1]) # weighted mean
    else:
        if parallelize:      
            # write fcn to take dst_group,mapping=mapping,structfun_data=structfun_data and update struct_fun_info
            def parallel_fcn(dst_grp, strfcn_datum, struct_fun_info = struct_fun_info):
                new = np.vstack((strfcn_datum,struct_fun_info.get(dst_grp,[0,0])))
                new_weight =  np.sum(new[:,1])
                struct_fun_info[dst_grp] = [np.nansum(np.prod(new,axis=1))/new_weight, new_weight] # update our dict, memory mapped
                return # return nothing
                                
            Parallel(**parallel_args)(delayed(parallel_fcn)(dst_grp,strfcn_datum,struct_fun_info = struct_fun_info) for dst_grp,strfcn_datum in it)
            struct_fun_info = {key:val[0] for key,val in struct_fun_info.items()} # drop the weights at the end


        else: # no parallelize
            for dst_grp,strfcn_datum in it: # iterate over generator/iterator/iterable
                struct_fun_info[dst_grp] = weighted_nanmean(strfcn_dat,struct_fun_info.get(dst_grp,[0,0])) # update value
   
    
    
    struct_fun_info = pd.Series(struct_fun_info)
    return struct_fun_info.sort_index() if sorted_output else output



        
    
def plot_sturucture_function(
    SF_dict,
    save_path  = None,
    save_args  = { 'bbox_inches':'tight', 'dpi':300},
    fig        = None,
    fig_args   = {'facecolor':'white'},
    _ax        = None,
    ax_args    = {'facecolor':'white'},
    cmap       = LinearSegmentedColormap.from_list("mycmap", [[0,0,1,1],[0,1,.5,1],[1,0,0,1],'yellow',[0,0,1,1]],N=13),
    linealpha  = .5, # this overloads any alphas in cmap if they exist... but works for setting the lines and markers to different opacities
    legend_opacity = 1, # you can manually set tue opacity of the legend markers, if this is None, it'll default to keep the one from the plotted lines
    xlim_scale = .9**np.array([1,-1]),
    ylim_scale = .9**np.array([1,-1]),
    xlabel     = 'separation distance [m]',
    ylabel     = '',
    title      = '',
    labelsize  = 14,
    titlesize  = 14,
    style      = 'normal',
    symlog_slim_args = {'frac':.02, 'linthresh_scale':1e-5},
    symlog_wide_args = {'frac':.2 , 'fixed_linthresh':lambda bnds: np.max(np.abs(bnds))*(10**-1.5)}, # placeholder as a function is evaluated to the actual value of that fcn call on the calculated y bounds of the plot
    slopes     = [], # list of [(slope, intercept, {kwargs}),...], if intercept is a scalar, it is the fraction across the x axis in log scale at which the line will cross the logscale center of the y axis
    default_slope_args = {'color':[.3, .3, .3], 'linestyle':'--','linewidth':1.25},
    leg1_args   = {'loc':'best'      ,'ncol':4,'fontsize':12,'facecolor':'white'},
    leg2_args   = {'loc':'lower center','ncol':6,'fontsize':12,'facecolor':'white'},
    zero_line = True,
    positive_marker_style = {'style':'o', 'markersize':3 , 'fillstyle':'full'},
    negative_marker_style = {'style':'o', 'markersize':7, 'fillstyle':'none'},
   ):
    
    """
    Can Help with plotting the output from structure_function
    """
    

    
    styles = ['normal', 'abs', 'linear', 'symlog slim','symlog wide'] # symlog cuts the middle down so it tries not to include data near 0, symlog wide has a wide moat for smooth transitions across 0
    
    if any(map(style.__contains__, ['normal','logarithmic'])) or (style == 'log'):    
        style = 'normal'
        scale = 'log'
        axline_style = 'loglog'
        axline_res   = None
    elif any(map(style.__contains__, ['abs'])):    
        style = 'abs'
        scale = 'log'
        axline_style = 'loglog'
        axline_res   = None
    elif any(map(style.__contains__, ['lin'])):    
        style = 'linear'
        scale = 'linear'
    elif any(map(style.__contains__, ['symlog slim','sym slim'])): 
        style = 'symlog slim'
        scale = 'symlog'
    elif any(map(style.__contains__, ['symlog wide','sym wide'])): 
        style = 'symlog wide'
        scale = 'symlog'

    else:
        raise ValueError('Unsupported plot style, choose from [normal/logarithmic, absolute value, linear, symlog slim, symlog wide]')


#     axline_style = 'linear'

            
    if fig is None:
        if _ax is None: 
            fig = personal.plots.default_figure(**fig_args)
            _ax = plt.axes(**ax_args)
    else:
        if _ax is None:
            _ax = fig.axes
            if len(_ax) == 0:
                _ax = plt.axes(**ax_args)
            else:
                _ax= _ax[-1] # attempt to recover the axes from the figure, will plot on the last axis in the list
        else:
            raise ValueError('If axis (_ax) is specified, do not specify a figure argument (fig), leave as default/None')

    
    l_tg   = len(SF_dict)
    colors = cmap(np.linspace(0, 1, l_tg+1)[0:l_tg])
    
           
    # Caclulate the real max/min VIA ITERATION OVER OUR GROUPS
    nxbounds     = [+np.inf, -np.inf]
    nybounds     = [+np.inf, -np.inf]
    nybounds_abs = [+np.inf, -np.inf]
    for tg in SF_dict.keys():
        n         = SF_dict[tg]             
        nabs      = np.abs(n)
        valid     = ~np.isinf(   n.values) * ~np.isnan(   n.values)
        valid_abs = ~np.isinf(nabs.values) * ~np.isnan(nabs.values)
        _nybounds     = personal.math.range_bounds(   n.values[valid   ])
        _nybounds_abs = personal.math.range_bounds(nabs.values[valid_abs])
        # compare new min/max to old and replace if necessary
        nybounds      = [ np.minimum(_nybounds[    0], nybounds[    0]), np.maximum(_nybounds[    1], nybounds[    1]) ]
        nybounds_abs  = [ np.minimum(_nybounds_abs[0], nybounds_abs[0]), np.maximum(_nybounds_abs[1], nybounds_abs[1]) ]
        
        _nxbounds     = personal.math.range_bounds(   n.index[valid    ])
        # compare new min/max to old and replace if necessary
        nxbounds      = [ np.minimum(_nxbounds[    0], nxbounds[    0]), np.maximum(_nxbounds[    1], nxbounds[    1]) ]
        
            
    for tg, color in zip(SF_dict.keys(),colors):
        n         = SF_dict[tg]             
        if style =='abs':
             nn,n = n,np.abs(n)
        else:
             nn,n = n,n
                
        n.dropna().plot(ax=_ax, c=color, label=tg          , style='-', alpha=linealpha) 
        n[nn>=0].plot(  ax=_ax, c=color, label='_nolegend_', **positive_marker_style)
        n[nn< 0].plot(  ax=_ax, c=color, label='_nolegend_', **negative_marker_style)
        
        
    if scale == 'symlog':
        ymax = np.max(np.abs(nybounds))
        if style == 'symlog slim':
            args = personal.plots.calculate_symlog_args(ymax, **symlog_slim_args)
            _ax.axhline( 1*args['linthresh'],color=[.5,.5,.5,.5],linestyle='--')
            _ax.axhline(-1*args['linthresh'],color=[.5,.5,.5,.5],linestyle='--')
        elif style == 'symlog wide':
            if personal.data_structures.isfunc(symlog_wide_args['fixed_linthresh']):
                symlog_wide_args['fixed_linthresh'] = symlog_wide_args['fixed_linthresh'](nybounds)
            args = personal.plots.calculate_symlog_args(ymax, **symlog_wide_args)
            _ax.axhline( 1*args['linthresh'],color=[.5,.5,.5,.5],linestyle='--')
            _ax.axhline(-1*args['linthresh'],color=[.5,.5,.5,.5],linestyle='--')
    else:
        args = {}
        
    # make  sure the x limits are set properly so axline doesn't trip on negative numbers
    _ax.set_xlim(nxbounds*np.array(xlim_scale))    
    if scale == 'log':
        _ax.set_ylim(np.abs(nybounds_abs) * np.array(ylim_scale))
    else:
        _ax.set_ylim(nybounds * np.power(np.array(ylim_scale), np.sign(nybounds)))
    xlim,ylim = _ax.get_xlim(), _ax.get_ylim() # the log scaling could fail if the data crosses 0, but it fails with only a warning, use these to track what the actual outcome for the limits is (needed for axline intercepts)

        
    for slope in slopes:
        if not personal.data_structures.isiterable(slope): slope = (slope,)
        slope = tuple(slope) + (None,)*(3-len(slope)) # fill in empty defaults with None
        slope,intercept,kwargs = slope
        if intercept is None:
            intercept = [xlim[0],ylim[0]]
        else:
            intercept = np.atleast_1d(np.array(intercept).squeeze()) # if is a scalar, now is is a numpy array, if was a numpy array is squeezed down from adding a dim, add in a dimension if was 1d
            l_int     = len(intercept)
            if l_int == 1: # assume it's the x fraction at which we pass through the center of the y axis (geometric means for scaling)
                if scale == 'log': # use geometric means
                    
                    intercept = [xlim[0] * (xlim[1]/xlim[0])**intercept, (ylim[0]*ylim[1])**.5]
                else: # keep logarithmic mean for x, but now use a linear one for y
                    intercept = [xlim[0] * (xlim[1]/xlim[0])**intercept, (ylim[0]+ylim[1])*.5]

            if l_int > 2:
                raise ValueError('Inavlid intercept coordinate of length > 2')
                
        if kwargs is None:
            kwargs = default_slope_args

        if scale == 'log':
            if personal.data_structures.get_package(slope,return_str=True) == 'fractions':
                slope = slope.limit_denominator() # for some reason it turns some simple fractions into nightmares and this helps fix that (e.g. 1/3 would become 6004799503160661/18014398509481984 )

            personal.plots.axline(_ax,slope = float(slope), ax_scale=axline_style, intercept=intercept,label='slope = '+str(slope), **kwargs)
        else:
            if len(slopes) > 0:
                raise ValueError('Reference slope lines only supported for loglog plots (logarithmic and aboslute value logarithmic) ')
    if zero_line: _ax.axhline(0,linestyle='--',color='k')

        
    _ax.set_xscale('log')
    _ax.set_yscale(scale, **args)
    
    leg   = _ax.get_legend_handles_labels()
    l_lab = len(leg[0])
        

    

    
    # plot 2nd legend (leg2) first since we're more likely to fix its position and then let the main legend with all the plotted lines float around if you selected location='best' for example
    if l_lab > l_tg:
        handles, labels = zip(*sorted(zip(leg[0][l_tg:], leg[1][l_tg:]), key=lambda t: float(Fraction(t[1].strip('slope = ')))))         # for some reason these unsort themselves, who knows why, so i force sort them by their slope, Fraction helps deal wit strings
        leg2  = _ax.legend(handles,labels, **leg2_args)
        _     = _ax.add_artist(leg2) # add legend to axis so it isn't overwritten (for some reason doensn't work)
        leg1  = _ax.legend(leg[0][ 0:l_tg], leg[1][ 0:l_tg], **leg1_args)
    else:
        leg1  = _ax.legend(leg[0][ 0:l_tg], leg[1][ 0:l_tg], **leg1_args)

    if legend_opacity is not None:
        for lh in leg1.legendHandles: lh.set_alpha(legend_opacity) # see https://stackoverflow.com/questions/12848808/set-legend-symbol-opacity-with-matplotlib
        
        
    _ax.set_xlabel(xlabel, fontsize=labelsize)
    _ax.set_title(title  , fontsize=titlesize)

    if save_path is not None:
        fig.savefig(save_path, **save_args)
        
    return fig, _ax
    
    