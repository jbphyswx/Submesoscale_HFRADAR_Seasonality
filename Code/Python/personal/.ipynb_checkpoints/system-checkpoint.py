import sys

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified
    prints the sizes of current items in memory'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def print_memory_item_sizes(local_items=None,n=-1,size_fcn=sys.getsizeof):
    ''' prints the memory item sizes of the items in the namespace of local_items
        to default to the caller namespace, use local_items= locals().items()'''
#     print([name for name,_ in local_items])
#     print(sorted(((name, sys.getsizeof(value)) for name, value in local_items)))
    for name, size in sorted(((name, size_fcn(value)) for name, value in local_items),
                             key= lambda x: -x[1])[:n]: print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
    
    
    
def get_size(obj, seen=None):
    """Recursively finds size of objects (sys.getsizeof only does the top level view, but could contain other objects inside like a dict)"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

