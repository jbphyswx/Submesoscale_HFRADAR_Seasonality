import scipy.spatial 
import numpy as np


def get_distances_from_coords(coord_list,distance_metric='euclidean',return_square=False,**kwargs):
    """
    coord_list is an array containing point locations, e.g. [(q11,q12,q13,...), (q21,q22,q23,...), ...]
    returns a list of the coordinate pairs with the distances between them
    
    metric can be a fcn
    """
    
    # the out coordinates are itertools.combinations(coord_list)
    out = scipy.spatial.distance.pdist(coord_list,metric=distance_metric,**kwargs)
    if return_square:
        return scipy.spatial.distance.squareform(out)
    return out
    
def haversine_np(u,v,a=6371*10**3):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1 = u[0]
    lat1 = u[1]
    lon2 = v[0]
    lat2 = v[1]
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    theta = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2 # angular distance

    c = 2 * np.arcsin(np.sqrt(theta))
    return c*a