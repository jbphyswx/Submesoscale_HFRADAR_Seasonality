"""
Plotting routines for python 

"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import animation, rcParams, cycler
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import sklearn

import warnings



import personal.data_structures


dpi=100                                                                      # default dpi
def default_figure(**kwargs): return plt.figure(figsize=(960/dpi,900/dpi),dpi=dpi,**kwargs)   # default figure commmand

def make_patch_spines_invisible(ax):
    """
    Shortcut for making spines invisible for an axis
    """
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
        
        
class Labeloffset():
    """
    Moves the label offset exponent to inside the axis label with a callback method so it will auto update if necessary
    see https://stackoverflow.com/questions/45760763/how-to-move-the-y-axis-scale-factor-to-the-position-next-to-the-y-axis-label
    see also https://peytondmurray.github.io/coding/fixing-matplotlibs-scientific-notation
    """
    separator = '   ' + r"$\times$" + ' '

    def __init__(self,  ax, label="", axis="y",separator=separator):
        self.axis = {"y":ax.yaxis, "x":ax.xaxis}[axis]
        self.label=label
        self.separator = separator
        ax.callbacks.connect(axis + 'lim_changed', self.update) # the axis calls this whenever its updated by (x,y)lim_changed methods
        ax.figure.canvas.draw()
        self.update([None])

    def update(self, event_axes):
        # Sets the actual axis label on an axis object (using its internal label and separator properties)
        # event_axes are the parent axes of this object that is calling an event
        # really is just a for callback formatting placeholder tho since we can access anything via self.axis

        fmt = self.axis.get_major_formatter()
        offset_text = fmt.get_offset()
        if offset_text: # if empty, returns false and we do nothin
            self.axis.offsetText.set_visible(False)
            self.axis.set_label_text(self.label + self.separator + fmt.get_offset() )
        else: # seems to write nothing
            self.axis.set_label_text(self.label)
            
def hist_to_bar(ax, bin_edges, count):
    """
    Converts histogram output to bar plot
    """
    
    width = np.diff(bin_edges)
    center = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return ax.bar(center,count,align='center',width=width)
            
        
        
# def animate_3D_data(data,num_frames=None, animate_fcn = ,animation_writer=,output_file=None,external_routines = ,fig_command=default_figure(),dpi=dpi):
#     """
#     Animates a 3D data plot by allowing you to step through it and animate.
#     if num_frames is none, automate to plot all frames
#     """
    
    
#     fig = fig_command()
    
    
#     anim = FuncAnimation(fig, animate, interval=interval, frames=indices[1:])
    
#     if output_file is not None: # save to disk
        
        
#     return 
    
    
class axline():

    """
    Draw a line based on its slope and y-intercept. Additional arguments are
    passed to the <matplotlib.lines.Line2D> constructor.
    adapted from https://stackoverflow.com/a/14348481/
    
    uses matplotlib.plot which returns a list of line2D objects (if linear returns just a single line object)
    
    """

    def __init__(self,ax=None, slope=1, intercept=None,extent='infinity',ax_scale='linear', resolution=None, *args, **kwargs):
        # intercept is an array/tuple/list of any point the line passes through, defaults to bottom left
        # ax_scale = 'linear','loglinear','linearlog','loglog','log'
        
        if ax is None: # default to current axes
            ax = plt.gca()
        self.ax = ax
            
        # if unspecified, get the current next line color from the axes
#         if not ('color' in kwargs or 'c' in kwargs):
#             kwargs.update({'color':ax._get_lines.color_cycle.next()})
        self.args   = args
        self.kwargs = kwargs
            
        self.xlim = ax.get_xlim()
        self.ylim = ax.get_ylim()
        if intercept is None:
            intercept = [self.xlim[0], self.ylim[0]] # default to lower left
            
        # solve for intercept if we're not already on the left boundary
        if ax_scale.lower() == 'linear':                           # linear x, linear y           -->     y  =       mx + b
            b = intercept[1] - slope*intercept[0]                  # b = y_0 - m*x_0
            def fcn(x,m=slope,b=b): return m*x + b
        elif ax_scale.lower() == 'loglinear':                      # linear x,    log y           --> log(y) =       mx + b
            b = np.log(intercept[1]) - slope*intercept[0]          # b = log(y_0) - m*x_0
            def fcn(x,m=slope,b=b): return np.exp(m*x + b)
        elif ax_scale.lower() == 'linearlog':                      #    log x, linear y           -->     y  = m*log(x) + b
            b = intercept[1] - slope*np.log(intercept[0])          # b = y_0 - m*log(x_0)
            def fcn(x,m=slope,b=b): return m*np.log(x) + b
        elif ax_scale.lower() == 'loglog':                         #    log x,    log y           --> log(y) = m*log(x) + b
            b = np.log(intercept[1]) - slope*np.log(intercept[0])  # b = log(y_0) - m*log(x_0)
            def fcn(x,m=slope,b=b): return np.exp(m*np.log(x) + b)
#             def fcn(x,m=slope,b=b): return m*x + b
        else:
            raise ValueError('unsuported axis scale (ax_scale): ' + str(ax_scale) + ', please choose from [linear, loglinear, linearlog, loglog]')
        self.fcn = fcn
        
        
        
        if resolution is None:
            if ax_scale.lower() == 'linear':
                self.resolution = 2
            elif ax_scale.lower() == 'loglinear': # linear x,    log y
                self.resolution = 100
            elif ax_scale.lower() == 'linearlog': #    log x, linear y 
                self.resolution = 100
            elif ax_scale.lower() == 'loglog':    #    log x,    log y
                self.resolution = 2
        x = np.linspace(self.xlim[0],self.xlim[1],num=self.resolution,endpoint=True)
        

        self.lines = ax.plot(x,self.fcn(x),*self.args,**self.kwargs)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        

        
#         ax.callbacks.connect(axis + 'lim_changed', self.update_lines) # the axis calls this whenever its updated by (x,y)lim_changed methods
        self.ax.callbacks.connect('xlim_changed', self.update_lines)
        self.ax.callbacks.connect('ylim_changed', self.update_lines)
        ax.figure.canvas.draw()
        self.update_lines([None])

           
    
    def update_lines(self, event_axes):
        # Sets the actual axis label on an axis object (using its internal label and separator properties)
        # event_axes are the parent axes of this object that is calling an event
        # really is just a for callback formatting placeholder tho since we can access anything via self.axis
        x_lim = self.ax.get_xlim()
        y_lim = self.ax.get_ylim()
        x = np.linspace(x_lim[0],x_lim[1],num=self.resolution,endpoint=True)
        

        for line in self.lines: self.ax.lines.remove(line) 
        self.lines = self.ax.plot(x,self.fcn(x),*self.args,**self.kwargs)
        

    
    
from holoviews.operation.datashader import datashade, spread, rasterize
import holoviews
import geoviews
import panel
import param

import geoviews.feature as gf

from cartopy import crs as ccrs
from geoviews import opts
  
def base_data_map_viewer(   data, 
                            kdims        = None, # key dimensions for the dataset
                            vdims        = None, # variable dimension for the dataset, figure out how 
                            plot_dims    = None, # dimensions for a plot, note for coastlines you must provide 'lon' first to get the correct bounding box
                            plot_vars    = None, # variables  for a plot, FIGURE HOW TO MAKE THIS INTO A SWITCHABLE MENU
                            plot_engine  = geoviews.QuadMesh, # could be geoview.Image, etc depending on the spacing we got (even spacing for example doesn't need QuadMesh). Note, for some reason, geoviews.Quadmesh shits the bed on some projections like ccrs.PlateCarree() or None, no idea why...
                            dynamic      = True,
                            projection   = None,
                            plot_coasts  = False,
                            coast_params = {'resolution':'10m', 'category':'physical', 'name':'coastline'}, # stick to this format
                            plotter      = holoviews.operation.datashader.rasterize, # seems to be fastest and most robust
                            precompute   = True,
                            cmap         = 'RdBu_r',
                            clim         = (None,None), # can also pass 'range' which owrks on first array in dataset so if you pass something with multiple variables won't work well...
                            framewise    = False,
                            colorbar     = True,
                            show_grid    = False,
                            interactive  = True,
                            output_size  = 300,
                            max_frames   = 10000, # more than this, consider down sampling or using dynamic loading (dynamic=True)
                            verbose      = False,
                            aspect       = None
                        ):
    """
    Viewer for data using geoviews/holoviews to create interactive viewers for data https://geoviews.org/
    To use sliders in jupyter lab, be sure the pyviz jupyter extension is installed and built, < jupyter labextension install @pyviz/jupyterlab_pyviz >, see https://geoviews.org/
    
    Default is for this to take <2-n>D data (especially geospatial) and plot it on a 2D plot (especially map)
    ... future support for making 1D line plots and other such averages could come in a separate function later
    ... ... If you don't put map axes, it should just plot the array, but it will be some form of a 2D plot
    
    data is some xarray dataset so cast to dataset if dask or numpy array using helper fcn
    
    # To do -- figure out how to use transforms (e.g. to plot data on a globe etc from just lat/lon data)
    """
    
    
    
    def print_verbose(x,verbose=verbose):
        if verbose: print(x)
        return
    
    # resolve defaults (may have been set to None, for example for the data_map_viewer fcn below)
    if cmap is None: cmap = 'RdBu_r'
    
    is_geospatial = personal.data_structures.get_package(plot_engine,return_str=True) == 'geoviews'
    if is_geospatial:
        views = geoviews
    else:
        views = holoviews
    
    if interactive:
        views.extension('bokeh','matplotlib')
    else:
        views.extension('matplotlib')
         
    views.output(size=output_size)
    views.output(max_frames=max_frames)
    


    data  = personal.data_structures.xr_resolve_type(data, otype='dataset') # ensure is dataset so you can pass in numpy/dask arrays
    extraneous_coords = [x for x in data.coords if x not in data.dims]
    if len(extraneous_coords) is not 0:
        print_verbose('dropping extraneous coords ' + str(extraneous_coords))
        data = data.drop(extraneous_coords) # coordinates not in dims seem to throw trouble if they're chunked, so we'll just drop them to be sure. Could call compute but that edits the underlying dataarray
    
    
    if kdims is None: kdims=list(data.dims)
    if vdims is None: vdims=list(data.data_vars)
    
        
    if projection is not None:
        xr_dataset = views.Dataset(data,kdims=kdims, vdims=vdims, crs=projection)
    else:
        try:
#             print(fail)
            print_verbose('trying to get projection from data...')
            projection = np.atleast_1d(data.crs)[0].to_cartopy()
            print_verbose('...retrieved projection ' + str(projection))
            xr_dataset = views.Dataset(data,kdims=kdims, vdims=vdims, crs=projection)
        except:
            print_verbose('...failed, continuing with no projection')
            xr_dataset = views.Dataset(data,kdims=kdims, vdims=vdims)   
    print_verbose(xr_dataset)
    
    if plot_dims is None: plot_dims=kdims[0:2] # default to plotting over first 2 dimensions
    if is_geospatial: # must be of form ['lon','lat'] for proper plotting
        if ('lat' in plot_dims[0].lower()) and ('lon' in plot_dims[1].lower()):
            print_verbose('flipping [lat,lon] to [lon,lat] for geospatial plotting')
            plot_dims = [plot_dims[1],plot_dims[0]] # opposite orientation is fine, otherwise just will fail in plotting probably, who knows
        
    if plot_vars is None: plot_vars=vdims
    plot_engine_output = xr_dataset.to(plot_engine, kdims=plot_dims, vdims=plot_vars, dynamic=dynamic)
    
    if clim is 'range':
        clim = tuple(personal.math.range_bounds(data[plot_vars[0]])) # wont work on dataset
        print_verbose('changed clim to ' + str(clim))
    elif clim is None:
        clim = (None,None)
        print_verbose('changed clim to ' + str(clim))


    if plot_coasts: # this whole loop speeds things up by cutting out only the part of the coastlines we need. Particularly faster at 10m resolution
        if not is_geospatial: warnings.warn('Plotting coastlines without geospatial plot_engine is unlikely to work correctly',RuntimeWarning)
        import shapely.geometry
        from shapely.ops import split, unary_union
        import cartopy.io.shapereader as shpreader
                
        dim_0 = data[plot_dims[0]]
        dim_1 = data[plot_dims[1]]
        dim_0_min,dim_0_max = np.min(dim_0),np.max(dim_0)
        dim_1_min,dim_1_max = np.min(dim_1),np.max(dim_1)
        bbox = shapely.geometry.box(*(dim_0_min, dim_1_min, dim_0_max, dim_1_max)) # draw the bounding box for this dataset

        shpfilename = shpreader.natural_earth(**coast_params)
        reader = shpreader.Reader(shpfilename)
        coast = unary_union([geom.geometry.intersection(bbox) for geom in reader.records()])
        
        if type(coast) is shapely.geometry.linestring.LineString: # you can't cast these to multiline strings? idk we just finna do it here
            out = plotter(plot_engine_output,precompute=precompute).options(cmap=cmap, colorbar=colorbar, clim=clim, show_grid=show_grid) * geoviews.Shape(coast) # plot coast 2nd so it's on top
            return out
        
        if len(shapely.geometry.multilinestring.MultiLineString(coast)) is 0:  # avoids throwing an error if there's no coastlines in the boxx
            print_verbose('No coastlines within given dimensions')
        else:
            print_verbose('Rendering with coastlines')
            out = plotter(plot_engine_output,precompute=precompute).options(cmap=cmap, colorbar=colorbar, clim=clim, show_grid=show_grid) * geoviews.Shape(coast) # plot coast 2nd so it's on top
            return out
        

    import geoviews.feature as gf
    print_verbose('Rendering')
    out =  plotter(plot_engine_output,precompute=precompute).options(cmap=cmap, colorbar=colorbar, clim=clim, show_grid=show_grid,aspect=aspect)
#     from holoviews.streams import Pipe, Buffer
    
#     variables= list(data.data_vars)
#     pipe = Pipe(data=variables[0])
#     plot_engine_output = holoviews.DynamicMap(plot_engine_output,streams=[pipe])
#     out = panel.Row(panel.panel(explorer.param, parameters=['variable']), plotter(plot_engine_output,precompute=precompute).options(cmap=cmap, colorbar=colorbar, clim=clim, show_grid=show_grid,aspect=aspect))
    return out

from collections import OrderedDict as odict

def data_map_viewer(data, **kwargs):
    """ kwargs last updated from base_data_map_viewer above on 09/29/2019, if an error in this list check for updated kwargs in base_data_map_viewer
    kdims        = None, # key dimensions for the dataset
    vdims        = None, # variable dimension for the dataset, figure out how 
    plot_dims    = None, # dimensions for a plot, note for coastlines you must provide 'lon' first to get the correct bounding box
    plot_vars    = None, # variables  for a plot, FIGURE HOW TO MAKE THIS INTO A SWITCHABLE MENU
    plot_engine  = geoviews.QuadMesh, # could be geoview.Image, etc depending on the spacing we got (even spacing for example doesn't need QuadMesh). Note, for some reason, geoviews.Quadmesh shits the bed on some projections like ccrs.PlateCarree() or None, no idea why...
    dynamic      = True,
    projection   = None,
    plot_coasts  = False,
    coast_params = {'resolution':'10m', 'category':'physical', 'name':'coastline'}, # stick to this format
    plotter      = holoviews.operation.datashader.rasterize, # seems to be fastest and most robust
    precompute   = True,
    cmap         = 'RdBu_r',
    clim         = (None,None), # can also pass 'range' which owrks on first array in dataset so if you pass something with multiple variables won't work well...
    framewise    = False,
    colorbar     = True,
    show_grid    = False,
    interactive  = True,
    output_size  = 300,
    max_frames   = 10000, # more than this, consider down sampling or using dynamic loading (dynamic=True)
    verbose      = False,
    aspect       = None
    
    You can pass any argument except coast_params as a dict with the same keys as data if data is a dict. 
    If you want different parameters like colormaps etc for each variable, the best way is to split your dataset into a dictionary and use this functionality to accomplish this
    """
    
    if type(data) in set([list,tuple]): # supported iterable types
        data = {'Dataset'+str(i): personal.data_structures.xr_resolve_type(data[i], otype='dataset') for i in range(len(data))}
    elif personal.data_structures.get_package(data,return_str=True) is 'xarray':
        data = {'dataset': personal.data_structures.xr_resolve_type(data, otype='dataset')}
    elif isinstance(data,dict):
        pass
    else:
        raise TypeError('Unsupported Input Type' + type(data))
    
    
    class sel_dset_var(): # inherit kwargs from data_map_viewer
        def varget(dset,var,**kwargs):
            in_args = kwargs
            for arg,argval in in_args.items():
                if arg not in ['coast_params']:
                    if isinstance(argval,dict): in_args[arg] = argval[dset] #convert dict listings to str8 up if we provided a dict
            out = base_data_map_viewer(data[dset][var], **kwargs)
            return  out# use above fcn...

    #Dashboard
    class Data_Explorer(param.Parameterized): # look up what the __init__ for param.Parameterized is later
#         use the idea that dfeault data <-- list(data.keys())[0]
        dset  = param.Selector(odict([(x,x) for x in data.keys()]),default=list(data.keys())[0]) # default to first dict
        varss = param.Selector(list(data[list(data.keys())[0]].data_vars),default=list(data[list(data.keys())[0]].data_vars)[0])

        # FUTURE: ATTEMPT TO IMPLEMENT SOMETHING THAT PREVENTS THE RESETTING OF SLIDERS FROM GEO/HOLOVIEWS NOT SET WITH THIS FCN

        @param.depends('dset',watch=True)
        def update_var(self):
            variables = list(data[self.dset].data_vars)
            self.param['varss'].objects = variables
            if self.varss in variables: # dont let reset variable
                self.param.set_param(varss=self.varss)
            else:
                self.param.set_param(varss=variables[0])

        def elem(self):return getattr(sel_dset_var,'varget')(self.dset,self.varss,**kwargs)
        def elem_yr(self):return getattr(self.elem(),'select')() # self.elem() -> varget(self.dset,self.varss) -> 'selector', call on () bc vargets return doesn't actually take any arguments...
        def viewable(self,**kwargs):return self.elem_yr
        
        


    explorer_test = Data_Explorer(name="")
    panel_out = panel.Row(panel.Param(explorer_test.param, expand_button=False),explorer_test.viewable())
    panel_out.servable()
    return panel_out
    
    
    
def calculate_symlog_args(ymax, frac=.3, linthresh_scale=1e-1, fixed_linthresh=None):
    """
    see - https://matplotlib.org/api/scale_api.html#matplotlib.scale.SymmetricalLogScale
    linthresh : Defines the range (-x, x), within which the plot is linear. This avoids having the plot go to infinity around zero.
    
    linscale  : This allows the linear range (-linthresh, linthresh) to be stretched relative to the logarithmic range. Its value is the number of decades to use for each half of the linear range.
                For example, when linscale == 1.0 (the default), the space used for the positive and negative halves of the linear range will be equal to one decade in the logarithmic range. 
    
    frac: fraction of plot taken up by the linear scale, so  frac = linscale / [np.log10(ymax) - np.log10(linthresh)] = linscale / np.log10(ymax/linthresh) = constant
    frac: fraction of plot taken up by the linear scale, so  frac = linscale / ([np.log10(ymax) - np.log10(linthresh)] + linscale) = linscale / [np.log10(ymax/linthresh) + linscale] = constant

    -- SPECS --
    specify either a linthresh_scale with fixed_linthresh = None or False
     or
    specify a fixed lintresh value
    
    linthresh_scale : you can specify a scaling for the linear threshold relative to ymax, default is 1/10 as large
    fixed_linthresh : if this is not 0, then the linear threshold is fixed to this value
    
    If fixed_threshold is not None or False, then we have specified some specific threshold value we desire to fit into the fraction of the plot that is linear...
    -- This scales the plot differently, since it leaves different amounts of y real estate to fit into the same fraction of logarithmic area.
    If fixed threshold is set
    -- Then threshold scales via linthresh_scale so the plot always looks the same, for any ymax
    
    Fix threshold works as long as the fixed threshold is less than the max of the data, so ymax > linthresh

    """
    ymax = np.abs(ymax) # make sure given value is positive
    
    if ((frac <0 ) or (frac > 1)):
        raise ValueError('fraction (frac) for the linear area must be between 0 and 1')
    
    if fixed_linthresh:
        linthresh = fixed_linthresh
    else:
        linthresh = linthresh_scale * ymax

    if linthresh > ymax:
        raise ValueError('linear threshold (fixed_linthresh) cannot be larger than the data maximum value (ymax) ')
        
    linscale  = frac/(1-frac) * np.log10(ymax/linthresh)
        
    return {'linthresh': linthresh, 'linscale':linscale}
        
        

    
    
def plot_2D_array_lines(data,
                        x                     = None,
                        y                     = None,
                        n                     = 'all',
                        fig                   = None,
                        fig_args              = {'facecolor':'white'},
                        _ax                   = None,
                        ax_args               = {'facecolor':'white'},
                        style                 = 'linear',
                        symlog_slim_args      = {'frac':.02, 'linthresh_scale':1e-5},
                        symlog_wide_args      = {'frac':.2 , 'fixed_linthresh':lambda bnds: np.max(np.abs(bnds))*(10**-1.5)}, # placeholder as a function is evaluated to the actual value of that fcn call on the calculated y bounds of the plot
                        extremum_start        = None,
                        y_fcn                 = lambda y,n,ind=None: np.geomspace( y[0], y[-1], n),
                        cmap                  = None,
                        repeat_start_at_end   = True,
                        repeat_coord_resolver = lambda x: x[-1] + (x[-1] - x[-2]),
                        x_pad_fcn             = lambda x: (x[0], x[-1]),
                        plot_normalized       = False,
                        label_formatter       = lambda val: f"{val:.2E}"
                                                                   
                        ):
    """
    Meant to take a 2D numpy array or xarray object and plot slices of it across dimension 0 as lines
    - Each index along the 1th axis is a new slice, the 0th axis is the axis along which we plot (i.e. x)
    - if data is an nparray it will be converted to an xarray with the coordinates x,y, which will default to just np.arange(len(axis)) for each axis if None
    
    - n is the # of lines you want plotted, will be ~ equally spaced in the data, default is just to plot all the lines.
    <> repeat_end_at_start = True repeats the first value again at the end, i.e. if you're plotting a cycle this is useful
    -- -- repeat_coord_resolver asks what do you want to do if you repeat the end coordinate to fix it to a new value so the cycle can plot. default is to add the same difference as the last and second to last to the last value, which should work for the default y values
    -- -- x_pad_fcn adjusts the xlimits from the range_boudns of x to be a little padded however you like. Default is no change.

    <> extremum uses the possible peak finding methods in personal.math.extreme_extremum_along_axis to help trim the data using <> r_fcn
    
    # Plot normalized normalizes the data to sit between 0 and 1 so you can compare the cycle easily across all slices
    
    """
                        
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



    if isinstance(data, np.ndarray):
        data = xr.DataArray(data, coords = [np.arange(np.shape[data][0]), np.arange(np.shape[data][1])], dims = ['x','y'])
    if x is None:
        x = data[data.dims[0]].values
    if y is None:
        y = data[data.dims[1]].values
    
    xdim,ydim = data.dims # the x and y dims on the plot, but their actual names (x is the dim we'll plot along, y is the dim our slices are taken from)

    if n == 'all':
        n = data.values.shape[1]
    y = data[ydim].values

    if extremum_start is not None:
        extr_ind,_,_     = personal.math.extreme_extremum_along_axis(data.values, axis = data.get_axis_num('r') , extremum = extremum_start)
    else:
        y_fcn=lambda y,n,ind=None: np.geomspace( y[0], y[-1], n)
        extr_ind,_,_     = None,None,None
    ys                   = y_fcn(y,n,extr_ind)
    ys_nearest, ysn_inds = personal.data_structures.nearest(ys, y)
    ysn_inds,valid       = np.unique(ysn_inds,return_index=True)
    ys_nearest           = ys_nearest[valid]


    if repeat_start_at_end:
        data  = xr.concat((data,data.isel({xdim:0})), dim=xdim)
        x_end = repeat_coord_resolver(x)
        x     = np.append(x, x_end)
        # data  = data.assign_coords({xdim:x})
    
    
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
    
#---------------------------------------------------------------------------------------#

    n            = data.values[:,ysn_inds].flatten()       
    nabs         = np.abs(n)
    valid        = ~np.isinf(   n) * ~np.isnan(   n)
    valid_abs    = ~np.isinf(nabs) * ~np.isnan(nabs)
    nybounds     = personal.math.range_bounds(   n[valid   ])
    nybounds_abs = personal.math.range_bounds(nabs[valid_abs])
    # nxbounds     = personal.math.range_bounds(x)
                
        
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
        
#---------------------------------------------------------------------------------------#

                              

     
    if x_pad_fcn is not None:
        _ax.set_xlim(*x_pad_fcn(x))
                        
    lc = len(ysn_inds)+1
    if cmap is None: cmap     = LinearSegmentedColormap.from_list("mycmap", [[1,0,0],[0,0,1]],N=lc)
    colors_here  = cmap(np.linspace(0, 1, lc)[0:lc-1])

    for i,ind in enumerate(ysn_inds[::-1]):
        E = data.isel({ydim:ind}).values
        if scale.lower() == 'log': E[E<0] = np.nan # only do if log scale to stop there being hella vertical lines
        if not plot_normalized:
            _ax.plot(x,E     ,color = colors_here[i], label=label_formatter(y[ind]) )
            _ax.set_yscale(scale, **args)

            
        else:
            E_norm = sklearn.preprocessing.minmax_scale(E,feature_range=(0,1),copy=True)
            _ax.plot(x,E_norm,color = colors_here[i], label=f"{y[ind]:.2E}" )
            _ax.set_yscale('linear'); _ax.set_ylim([0,1])

    leg = _ax.legend(loc='upper center',ncol=8, bbox_to_anchor=(.5, -0.05))
                        
    return fig, _ax


    
    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """
    Truncate a colormap for your useful pleasure, e.g. to remove edge colors that are problematic for you. Use minval and maxval as needed
    see https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


