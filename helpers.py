import os
import sys
import flopy

def ws1_mod(sim):
    """
    Used to load the Flopy GitHub intro model by passing in your own sim object.


    Parameters
    ----------
    sim : object
       simulation object created with flopy.mf6.MFSimulation
    
    Methods
    ----------
    None
    
    Returns
    ----------
    None
    """    
    #ws = './model'
    name = 'MySim'
    #sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name='mf6')
    mod_name = 'MyModel'
    tdis = flopy.mf6.ModflowTdis(sim)
    print('building tdis package')
    ims = flopy.mf6.ModflowIms(sim)
    print('building ims package')
    gwf = flopy.mf6.ModflowGwf(sim, modelname=mod_name, save_flows=True)
    print('building gwf package')
    dis = flopy.mf6.ModflowGwfdis(gwf, nrow=10, ncol=10)
    print('building dis package')
    ic = flopy.mf6.ModflowGwfic(gwf)
    print('building ic package')
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True)
    print('building npf package')
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, 0), 1.],
                                                           [(0, 9, 9), 0.]])
    print('building chd package')
    budget_file = mod_name + '.bud'
    head_file = mod_name + '.hds'
    oc = flopy.mf6.ModflowGwfoc(gwf,
                                budget_filerecord=budget_file,
                                head_filerecord=head_file,
                                saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
    print('building oc package')
    return
    
def ws1_mod_trans(sim,tdis):
    """
    Used to load a transient version of the Flopy GitHub intro model by passing in your own sim and tdis object.


    Parameters
    ----------
    sim : object
       simulation object created with flopy.mf6.MFSimulation
    tdis : object
       simulation timing object created with flopy.mf6.ModflowTdis
    Methods
    ----------
    None
    
    Returns
    ----------
    None
    """    
    #ws = './model'
    name = 'MySim'
    #sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name='mf6')
    mod_name = 'MyModel'
    #tdis = flopy.mf6.ModflowTdis(sim)
    #print('building tdis package')
    ims = flopy.mf6.ModflowIms(sim)
    print('building ims package')
    gwf = flopy.mf6.ModflowGwf(sim, modelname=mod_name, save_flows=True)
    print('building gwf package')
    dis = flopy.mf6.ModflowGwfdis(gwf, nrow=10, ncol=10)
    print('building dis package')
    ic = flopy.mf6.ModflowGwfic(gwf)
    print('building ic package')
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True)
    print('building npf package')
    sto = flopy.mf6.ModflowGwfsto(gwf,save_flows=True,iconvert=1,
                                  ss=1.0E-05,sy=0.3,steady_state={0: True},
                                  transient={1: True})
    print('building sto package')
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, 0), 1.],
                                                           [(0, 9, 9), 0.]])
    print('building chd package')
    budget_file = mod_name + '.bud'
    head_file = mod_name + '.hds'
    oc = flopy.mf6.ModflowGwfoc(gwf,
                                budget_filerecord=budget_file,
                                head_filerecord=head_file,
                                saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
    print('building oc package')
    return
    
def ws1_mod_trans2(sim,tdis,ims):
    """
    Used to load a transient version of the Flopy GitHub intro model by passing in your own sim and tdis object.


    Parameters
    ----------
    sim : object
       simulation object created with flopy.mf6.MFSimulation
    tdis : object
       simulation timing object created with flopy.mf6.ModflowTdis
    ims : object
        simulation solver object created with flopy.mf6.ModflowIms
    
    Methods
    ----------
    None
    
    Returns
    ----------
    None
    """    
    #ws = './model'
    name = 'MySim'
    #sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name='mf6')
    mod_name = 'MyModel'
    #tdis = flopy.mf6.ModflowTdis(sim)
    #print('building tdis package')
    #ims = flopy.mf6.ModflowIms(sim)
    #print('building ims package')
    gwf = flopy.mf6.ModflowGwf(sim, modelname=mod_name, save_flows=True)
    print('building gwf package')
    dis = flopy.mf6.ModflowGwfdis(gwf, nrow=10, ncol=10)
    print('building dis package')
    ic = flopy.mf6.ModflowGwfic(gwf)
    print('building ic package')
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True)
    print('building npf package')
    sto = flopy.mf6.ModflowGwfsto(gwf,save_flows=True,iconvert=1,
                                  ss=1.0E-05,sy=0.3,steady_state={0: True},
                                  transient={1: True})
    print('building sto package')
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, 0), 1.],
                                                           [(0, 9, 9), 0.]])
    print('building chd package')
    budget_file = mod_name + '.bud'
    head_file = mod_name + '.hds'
    oc = flopy.mf6.ModflowGwfoc(gwf,
                                budget_filerecord=budget_file,
                                head_filerecord=head_file,
                                saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
    print('building oc package')
    return
    
def ws1_mod_trans3(sim,tdis,ims,gwf,modnam):
    """
    Used to load a transient version of the Flopy GitHub intro model by passing in your own sim and tdis object.


    Parameters
    ----------
    sim : object
       simulation object created with flopy.mf6.MFSimulation
    tdis : object
       simulation timing object created with flopy.mf6.ModflowTdis
    ims : object
       simulation solver object created with flopy.mf6.ModflowIms
    gwf : object
       simulation model object created with flopy.mf6.ModflowIms
    modnam: string
       the user specified model name string
    Methods
    ----------
    None
    
    Returns
    ----------
    None
    """    
    #ws = './model'
    name = 'MySim'
    #sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name='mf6')
    mod_name = modnam
    #tdis = flopy.mf6.ModflowTdis(sim)
    #print('building tdis package')
    #ims = flopy.mf6.ModflowIms(sim)
    #print('building ims package')
    gwf = gwf
    #print('building gwf package')
    dis = flopy.mf6.ModflowGwfdis(gwf, nrow=10, ncol=10)
    print('building dis package')
    ic = flopy.mf6.ModflowGwfic(gwf)
    print('building ic package')
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True)
    print('building npf package')
    sto = flopy.mf6.ModflowGwfsto(gwf,save_flows=True,iconvert=1,
                                  ss=1.0E-05,sy=0.3,steady_state={0: True},
                                  transient={1: True})
    print('building sto package')
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, 0), 1.],
                                                           [(0, 9, 9), 0.]])
    print('building chd package')
    budget_file = mod_name + '.bud'
    head_file = mod_name + '.hds'
    oc = flopy.mf6.ModflowGwfoc(gwf,
                                budget_filerecord=budget_file,
                                head_filerecord=head_file,
                                saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
    print('building oc package')
    return
    
def ws3_mod1(sim,gwf):
    """
    Used to load a SS version of the Flopy GitHub intro model by passing in your own objects.

    Parameters
    ----------
    sim : object
       simulation object created with flopy.mf6.MFSimulation
    tdis : object
       simulation timing object created with flopy.mf6.ModflowTdis
    ims : object
       simulation solver object created with flopy.mf6.ModflowIms
    gwf : object
       simulation model object created with flopy.mf6.ModflowIms
    chd : object
       constant head boundary object created with flopy.mf6.ModflowChd
    Methods
    ----------
    None
    
    Returns
    ----------
    None
    """    
    import matplotlib.pyplot as plt
    name = gwf.name
    print('building ic package')
    ic = flopy.mf6.ModflowGwfic(gwf)
    print('building npf package')
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True)

    budget_file = name + '.bud'
    head_file = name + '.hds'
    print('building oc package')
    oc = flopy.mf6.ModflowGwfoc(gwf,
                                budget_filerecord=budget_file,
                                head_filerecord=head_file,
                                saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
    print('writing simualtion')
    sim.write_simulation()
    print('run simualtion')
    sim.run_simulation()
    print('extracting heads')
    head = gwf.output.head().get_data()
    print('extracting cell-by_cell flows from budget')
    bud = gwf.output.budget()
    spdis = bud.get_data(text='DATA-SPDIS')[0]
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)
    print('building Mapview and plotting')
    fig,ax = plt.subplots(figsize=(8,5),subplot_kw={'aspect':'equal'})
    pmv = flopy.plot.PlotMapView(modelgrid=gwf.modelgrid, ax=ax)
    pmv_head = pmv.plot_array(head)
    pmv.plot_grid(colors='white')
    pmv.plot_vector(qx, qy, normalize=True, color="white")
    plt.colorbar(pmv_head, aspect=30)
    return()

def ws3_mod1_trans(sim,gwf):
    """
    Used to load a SS version of the Flopy GitHub intro model by passing in your own objects.

    Parameters
    ----------
    sim : object
       simulation object created with flopy.mf6.MFSimulation
    tdis : object
       simulation timing object created with flopy.mf6.ModflowTdis
    ims : object
       simulation solver object created with flopy.mf6.ModflowIms
    gwf : object
       simulation model object created with flopy.mf6.ModflowIms
    chd : object
       constant head boundary object created with flopy.mf6.ModflowChd
    Methods
    ----------
    None
    
    Returns
    ----------
    None
    """    
    import matplotlib.pyplot as plt
    name = gwf.name
    print('building ic package')
    ic = flopy.mf6.ModflowGwfic(gwf)
    print('building npf package')
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True)
    print('building sto package')
    sto = flopy.mf6.ModflowGwfsto(gwf,save_flows=True,iconvert=1,
                                  ss=1.0E-05,sy=0.3,steady_state={0: True},
                                  transient={1: True})
    budget_file = name + '.bud'
    head_file = name + '.hds'
    print('building oc package')
    oc = flopy.mf6.ModflowGwfoc(gwf,
                                budget_filerecord=budget_file,
                                head_filerecord=head_file,
                                saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
    print('writing simualtion')
    sim.write_simulation()
    print('run simualtion')
    sim.run_simulation()
    return()
    
def ws5_model1(ws5,gis_f,model_f,plots_f):
    import os
    import sys
    import shutil
    import platform
    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import flopy
    from flopy.discretization import VertexGrid
    from flopy.utils import Raster
    from flopy.utils import GridIntersect
    from flopy.utils.gridgen import Gridgen
    sim_name = "MySim" 
    sim = flopy.mf6.MFSimulation(sim_name=sim_name, 
                                exe_name="mf6",
                                verbosity_level=1,
                                sim_ws=model_f) 
    
    model_name = 'flow' 
    gwf = flopy.mf6.ModflowGwf(sim, 
                            modelname=model_name, 
                            save_flows=True, 
                            newtonoptions="under_relaxation")
    
    shp_path = os.path.join('files','disv_shapefiles') # path to shapefiles for this example
    flist = [x for x in os.listdir(shp_path)] # create a list of all the shapefiels
    for file in flist:
        shutil.copyfile(os.path.join(shp_path,file),os.path.join(gis_f,file)) 
    
    nlay = 3
    nrow = 34
    ncol = 44
    delr = delc = 1280.0
    botm = np.zeros((nlay, nrow, ncol), dtype=np.float32)
    top = np.zeros((1, nrow, ncol), dtype=np.float32)
    idom = np.ones((nlay, nrow, ncol), dtype=np.float32)
    botm[0, :, :] = 390.0
    botm[1,:,:] = 380.0
    botm[2,:,:] = -170.0
    top[0,:,:] = 460.0
    
    
    # Note we start with a structured DIS grid despite aiming for a DISV grid.
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        xorigin=729425,
        yorigin=947000,
        length_units='meters',
        angrot=0,
        idomain = idom
    )
    dis.export(os.path.join(gis_f,'disv.shp'))
    
    from shapely.geometry import Polygon
    g = Gridgen(dis)
    dam = os.path.join(gis_f,"dam_buffer")
    chanel = os.path.join(gis_f,"my_channels")
    tsf = os.path.join(gis_f,"tsf_buffer")
    wels = os.path.join(gis_f,"Wells_buffered")
    pit1500 = os.path.join(gis_f,"pits_buffer_1500")
    pit1000 = os.path.join(gis_f,"pits_buffer_1000")
    pit500 = os.path.join(gis_f,"pits_buffer_500")
    mod_bnd = os.path.join(gis_f,"model_bounds")
    act_dom = os.path.join(gis_f,"model_bounds_poly")
    
    g.add_refinement_features(chanel, "line", 3, layers=[0,1,2])
    g.add_refinement_features(wels, "polygon", 3, layers=[0,1,2])
    g.add_refinement_features(dam, "polygon", 3, layers=[0,1,2])
    g.add_refinement_features(tsf, "polygon", 3, layers=[0,1,2])
    g.add_refinement_features(pit1500, "polygon", 3, layers=[0,1,2])
    g.add_refinement_features(mod_bnd, "line", 3, layers=[0,1,2])
    g.add_refinement_features(pit1000, "polygon", 4, layers=[0,1,2])
    g.add_refinement_features(pit500, "polygon", 5, layers=[0,1,2])
    g.add_active_domain(act_dom,layers=[0,1,2])
    g.build()
    
    grd_files = [file for file in os.listdir('.') if file.startswith("qtgrid")]
    for file in grd_files:
        shutil.copyfile(file,os.path.join(gis_f,file))
    
    gridprops_vg = g.get_gridprops_vertexgrid()
    vgrid = flopy.discretization.VertexGrid(**gridprops_vg)
    fig,ax = plt.subplots(figsize=(12,12))
    vgrid.plot(ax=ax)
    ax.set_ylabel('Northing')
    plt.title('Model Grid')
    figname = os.path.join(plots_f,'model_grid.png') 
    fig.savefig(figname,dpi=300)
    figname = os.path.join(plots_f,'model_grid.pdf')
    fig.savefig(figname,dpi=300) 
    
    gridprops_disv = g.get_gridprops_disv()
    
    sim = flopy.mf6.MFSimulation(sim_name=sim_name, 
                            exe_name="mf6",
                            verbosity_level=1,
                            sim_ws=model_f) 
    start_date = "2023-12-31" 
    dates = pd.date_range('2024-01-01','2025-01-01', freq='MS').tolist()
    perlens = [(dates[x]-dates[x-1]).days for x in range(1,len(dates))]
    stp = 1 
    _ = [(x,stp,1) for x in perlens]
    pdata = [(1,1,1), *_]
    perlens = [x[0] for x in pdata]
    numper = len(pdata)
    
    dates = [pd.to_datetime(start_date),*dates] 
    # need to drop the last one. This is different to what we did previously. Why?
    dates = dates[:-1]
    df = pd.DataFrame() 
    df['Date'] = dates 
    df['SP'] = range(1,len(dates)+1) 
    df['Flopy_SP'] = range(len(dates)) 
    df['Incremental'] = perlens
    df['Cumulative'] = np.cumsum(perlens) 
    df.to_csv(os.path.join(model_f,'model_timing.csv'),index=None) 
    modtime_df = df.copy()
    
    tdis = flopy.mf6.ModflowTdis(sim,
                                time_units='days',
                                nper=numper,
                                perioddata=pdata,
                                start_date_time=start_date) 
    
    ims = flopy.mf6.ModflowIms(sim, complexity='MODERATE', 
                            csv_inner_output_filerecord='inner.csv', 
                            csv_outer_output_filerecord='outer.csv', 
                            outer_maximum=500, 
                            inner_maximum=500, 
                            outer_dvclose=0.01, 
                            inner_dvclose=0.001) 
                            
    model_name= 'flow'
    gwf = flopy.mf6.ModflowGwf(sim, modelname=model_name, save_flows=True, newtonoptions="under_relaxation")
    disv = flopy.mf6.ModflowGwfdisv(gwf,angrot=0,length_units="METERS", **gridprops_vg)
    grid_path = os.path.join(ws5,'Gridgen')
    if os.path.exists(grid_path): 
        shutil.rmtree(grid_path)
        os.mkdir(grid_path)
    else:
        os.mkdir(grid_path) 
    flist = [] 
    for pref in ['qtg', 'quadtree', '_gridgen',]: 
        temp_list = [x for x in os.listdir() if x.startswith(pref)] 
        flist = [*flist,*temp_list] 
    for file in flist:
        shutil.move(file,grid_path) 
    
    topo_fyl = os.path.join('.','files','filled_dem.tif')
    rio1 = Raster.load(topo_fyl)
    mg=gwf.modelgrid
    top_data = rio1.resample_to_grid(mg, band=rio1.bands[0], method="nearest")
    top_data[top_data>450.0]=450.0
    
    def scale_me(mx1,mn1,mx2,mn2,x):
        r1 = mx1-mn1
        r2 = mx2-mn2
        return((((x-mn1)*r2)/r1)+mn2)
    vf = np.vectorize(scale_me) # We vectorize the function just to make it quicker
    
    tmax = np.max(top_data)
    tmin = np.min(top_data)
    l1max = 60.0
    l1min = 30.0
    l1range = l1max-l1min
    l1_thickness = vf(tmax,tmin,l1max,l1min,top_data) # this is an array of thickness for layer 1 directly correlated with elevation
    
    l2max = 65.0
    l2min = 50.0
    l2_thickness = vf(tmax,tmin,l2max,l2min,top_data) # this is an array of thickness for layer 2 also directly correlated with elevation
    
    new_botms = np.ones_like(mg.botm)
    new_botms[0] = top_data - l1_thickness 
    new_botms[1] = new_botms[0] - l2_thickness
    new_botms[2] = new_botms[1]-370.0 
    l3_thickness = new_botms[1] - new_botms[2] 
    
    disv.botm = new_botms
    disv.top = top_data
    mg = gwf.modelgrid
    
    kx_layer_prop = [0.5,0.05,0.005] # m/d 
    kx_array = np.ones_like(mg.botm)
    for i,j in enumerate(kx_layer_prop):
        kx_array[i]=j
    #kv_layer_prop = [0.08,0.05,0.0005] # m/d 
    kv_layer_prop = kx_layer_prop
    kv_array = np.ones_like(mg.botm)
    for i,j in enumerate(kv_layer_prop):
        kv_array[i]=j
    
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        xt3doptions=False,
        pname="npf",
        save_flows=True,
        thickstrt=True,
        icelltype = 1,
        k= kx_array, # ading a list of names here automatically triggers the external file
        k33=kv_array,
    )
    
    sy_layer_prop = [0.01,0.02,0.001]
    syarray1 = np.ones_like(mg.botm)
    for i,j in enumerate(sy_layer_prop):
        syarray1[i]=j
    ssarray1 = np.ones_like(mg.botm)*1.0E-5
    ssarray1[1] =  1.0E-6
    ssarray1[2] =  1.0E-7
    ictype = np.ones_like(mg.botm)
    ictype[0] = 1
    ictype[1] = 1
    
    sto = flopy.mf6.ModflowGwfsto(
        gwf,
        pname="sto",
        save_flows=True,
        iconvert=ictype,
        ss=ssarray1,
        sy=syarray1,
        steady_state={0: True},
        transient={1: True},
    )
    
    # Lets get our boundary cells into groups we already have chanel, and mod_bnd loaded from gridgen
    # but we will add then again here just for completness
    chanel = os.path.join(gis_f,"my_channels.shp")
    ghb_north = os.path.join(gis_f,"ghb_north.shp")
    ghb_south = os.path.join(gis_f,"ghb_south.shp")
    lamarahoue = os.path.join(gis_f,"La_Marahoue_river_boundary.shp")
    bandamrouge = os.path.join(gis_f,"Bandam_Rouge_river_boundary.shp")
    yani = os.path.join(gis_f,"Yani_river_boundary.shp")
    grid = os.path.join(gis_f,"qtgrid.shp")
    
    import geopandas as gpd
    
    def get_bnodes(shpfyl): # works with single and multiple line strings
        ix = GridIntersect(mg, method="vertex")
        poly = gpd.read_file(shpfyl).geometry
        if len(poly)==1:
            return(ix.intersect(poly[0]).cellids)
        else:
            ls = []
            for item in poly:
                nums = ix.intersect(item).cellids
                ls = [*ls,*nums]
            return(np.asarray(ls))
    
    yani_nodes = get_bnodes(yani)
    bandamrouge_nodes = get_bnodes(bandamrouge)
    lamarahoue_nodes = get_bnodes(lamarahoue)
    ghb_south_nodes = get_bnodes(ghb_south)
    ghb_north_nodes = get_bnodes(ghb_north)
    chanel_nodes = get_bnodes(chanel)
    
    # get our range mapping function
    def scale_me(mx1,mn1,mx2,mn2,x):
        r1 = mx1-mn1
        r2 = mx2-mn2
        return((((x-mn1)*r2)/r1)+mn2)
    vf = np.vectorize(scale_me) # We vectorize the function just to make it quicker
    
    # get topo for ghb_north only
    # we are going to say what the depth to water is along this boundary by scaling it
    # we know there is a peak in the middle so lets find that first
    topo = [mg.top[x] for x in ghb_north_nodes]
    max = np.max(topo)
    max_id = topo.index(max)
    # okay so now we need to split the boundary into two different ranges
    range1 = topo[0:max_id+1]
    range2 = topo[max_id+1:]
    # now we create our heads for range1
    r1min = np.min(range1)
    r1max = max
    dtw_min = 2.0
    dtw_max = 40.0
    r1_dtw = vf(r1max,r1min,dtw_max,dtw_min,range1)
    r1heads = range1-r1_dtw
    # repeat for range2
    r2min = np.min(range2)
    r2max = max
    dtw_min = 2.0
    dtw_max = 40.0
    r2_dtw = vf(r2max,r2min,dtw_max,dtw_min,range2)
    r2heads = range2-r2_dtw
    # now unpack into a new list 
    ghb_north_heads = [*r1heads,*r2heads]*3 # will need to repeat for three layers
    
    node_tups = [(i,j) for i in range(3) for j in ghb_north_nodes]
    ghb1_pdata = [(item,ghb_north_heads[i],mg.cell_thickness[item[0]][item[1]]*kx_layer_prop[item[0]],'ghb1') for i,item in enumerate(node_tups)]
    ghb1_period={}
    ghb1_period[0] = ghb1_pdata
    ghb1 = flopy.mf6.ModflowGwfghb(gwf,boundnames=True,save_flows=True, maxbound=len(ghb1_pdata),\
                                stress_period_data=ghb1_period,pname='ghb1',
                                filename="{}_1.ghb".format(model_name),)
    
    # Setup obs arrays for drn
    obs1_recarray = {
        "ghb1_obs.csv": [
            ("ghb1", "GHB", 'ghb1')]
    }
    ghb1.obs.initialize(
        filename="{}_1.ghb.obs".format(model_name),
        digits=10,
        print_input=True,
        continuous=obs1_recarray,
    )
    
    mg=gwf.modelgrid
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    pmv = flopy.plot.PlotMapView(modelgrid=mg)
    pmv.plot_grid(ax=ax, lw=0.3, color="grey", alpha=0.3)
    pmv.plot_bc(name='ghb1',package=ghb1)
    ax.set_title('GHB Cells North')
    ax.set_xlabel('Eastings')
    ax.set_ylabel('Northings')
    ax.ticklabel_format(style='plain') #  gets rid of the exponent offsets on the axis
    plt.tight_layout()
    
    # get topo for ghb_south only
    # we are going to say what the depth to water is along this boundary by scaling it
    # we look for a peak in the middle so lets find that first
    topo = [mg.top[x] for x in ghb_south_nodes]
    max = np.max(topo)
    max_id = topo.index(max)
    # okay so now we need to split the boundary into two different ranges
    range1 = topo[0:max_id+1]
    range2 = topo[max_id+1:]
    # now we create our heads for range1
    r1min = np.min(range1)
    r1max = max
    dtw_min = 2.0
    dtw_max = 30.0
    r1_dtw = vf(r1max,r1min,dtw_max,dtw_min,range1)
    r1heads = range1-r1_dtw
    # repeat for range2
    r2min = np.min(range2)
    r2max = max
    dtw_min = 2.0
    dtw_max = 30.0
    r2_dtw = vf(r2max,r2min,dtw_max,dtw_min,range2)
    r2heads = range2-r2_dtw
    # now unpack into a new list 
    ghb_south_heads = [*r1heads,*r2heads]*3 # will need to repeat for three layers
    
    # now we can start building our boundary condition
    node_tups = [(i,j) for i in range(3) for j in ghb_south_nodes]
    ghb2_pdata = [(item,ghb_south_heads[i],mg.cell_thickness[item[0]][item[1]]*kx_layer_prop[item[0]],'ghb2') for i,item in enumerate(node_tups)]
    ghb2_period={}
    ghb2_period[0] = ghb2_pdata
    ghb2 = flopy.mf6.ModflowGwfghb(gwf,boundnames=True,save_flows=True, maxbound=len(ghb2_pdata),\
                                stress_period_data=ghb2_period,pname='ghb2',
                                filename="{}_2.ghb".format(model_name),)
    
    # Setup obs arrays for drn
    obs2_recarray = {
        "ghb2_obs.csv": [
            ("ghb2", "GHB", "ghb2")]
    }
    ghb2.obs.initialize(
        filename="{}_2.ghb.obs".format(model_name),
        digits=10,
        print_input=True,
        continuous=obs2_recarray,
    )
    
    mg=gwf.modelgrid
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    pmv = flopy.plot.PlotMapView(modelgrid=mg)
    pmv.plot_grid(ax=ax, lw=0.3, color="grey", alpha=0.3)
    pmv.plot_bc(name='ghb2',package=ghb2)
    ax.set_title('GHB Cells South')
    ax.set_xlabel('Eastings')
    ax.set_ylabel('Northings')
    ax.ticklabel_format(style='plain') #  gets rid of the exponent offsets on the axis
    plt.tight_layout()
    
    # River and chanel drain boundaries should be a bit easier 
    # because we will just assume drain elevation is about 1m below topography
    # Lets start with the drains for the Yani. Recall we only need these in layer 1
    lay = 0
    # river is not full width of cell
    # assume width of river is 20 m x cell length 320 m gives
    a = 20*320
    # initial conductance estimate is K*A
    cond0 = kx_layer_prop[0]*a
    drn1_pdata = [((lay,node),mg.top[node]-1,cond0,"yani") for node in yani_nodes]
    drn1_period={}
    drn1_period[0]=drn1_pdata
    drn1 = flopy.mf6.ModflowGwfdrn(gwf,boundnames=True,save_flows=True, maxbound=len(drn1_pdata),\
                                stress_period_data=drn1_period,pname='drn1',
                                filename="{}_1.drn".format(model_name),)
    # Setup obs arrays for drn
    obs3_recarray = {
        "drn_yani_obs.csv": [
            ("drn1", "DRN", "yani")]
    }
    drn1.obs.initialize(
        filename="{}_1.drn.obs".format(model_name),
        digits=10,
        print_input=True,
        continuous=obs3_recarray,
    )
    
    mg=gwf.modelgrid
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    pmv = flopy.plot.PlotMapView(modelgrid=mg)
    pmv.plot_grid(ax=ax, lw=0.3, color="grey", alpha=0.3)
    pmv.plot_bc(name='yani',package=drn1,color='red')
    ax.set_title('Drain Cells Yani River')
    ax.set_xlabel('Eastings')
    ax.set_ylabel('Northings')
    ax.ticklabel_format(style='plain') #  gets rid of the exponent offsets on the axis
    plt.tight_layout()
    
    drn2_pdata = [((lay,node),mg.top[node]-1,cond0,"lamarahoue") for node in lamarahoue_nodes]
    drn2_period={}
    drn2_period[0]=drn2_pdata
    drn2 = flopy.mf6.ModflowGwfdrn(gwf,boundnames=True,save_flows=True, maxbound=len(drn2_pdata),\
                                stress_period_data=drn2_period,pname='drn2',
                                filename="{}_2.drn".format(model_name),)
    # Setup obs arrays for drn
    obs4_recarray = {
        "drn_lamarahoue_obs.csv": [
            ("drn2", "DRN", "lamarahoue")]
    }
    drn2.obs.initialize(
        filename="{}_2.drn.obs".format(model_name),
        digits=10,
        print_input=True,
        continuous=obs4_recarray,
    )
    
    mg=gwf.modelgrid
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    pmv = flopy.plot.PlotMapView(modelgrid=mg)
    pmv.plot_grid(ax=ax, lw=0.3, color="grey", alpha=0.3)
    pmv.plot_bc(name='lamarahoue',package=drn2, color = 'red')
    ax.set_title('Drain Cells La Marahoue River')
    ax.set_xlabel('Eastings')
    ax.set_ylabel('Northings')
    ax.ticklabel_format(style='plain') #  gets rid of the exponent offsets on the axis
    plt.tight_layout()
    
    drn3_pdata = [((lay,node),mg.top[node]-1,cond0,"bandamrouge") for node in bandamrouge_nodes]
    drn3_period={}
    drn3_period[0]=drn3_pdata
    drn3 = flopy.mf6.ModflowGwfdrn(gwf,boundnames=True,save_flows=True, maxbound=len(drn3_pdata),\
                                stress_period_data=drn3_period,pname='drn3',
                                filename="{}_3.drn".format(model_name),)
    # Setup obs arrays for drn
    obs5_recarray = {
        "drn_bandamrouge_obs.csv": [
            ("drn3", "DRN", "bandamrouge")]
    }
    drn3.obs.initialize(
        filename="{}_3.drn.obs".format(model_name),
        digits=10,
        print_input=True,
        continuous=obs5_recarray,
    )
    
    mg=gwf.modelgrid
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    pmv = flopy.plot.PlotMapView(modelgrid=mg)
    pmv.plot_grid(ax=ax, lw=0.3, color="grey", alpha=0.3)
    pmv.plot_bc(name='bandamrouge',package=drn3, color = 'red')
    ax.set_title('Drain Cells Bandam Rouge River')
    ax.set_xlabel('Eastings')
    ax.set_ylabel('Northings')
    ax.ticklabel_format(style='plain') #  gets rid of the exponent offsets on the axis
    plt.tight_layout()
    
    drn4_pdata = [((lay,node),mg.top[node]-0.5,cond0,"channel") for node in chanel_nodes]
    drn4_period={}
    drn4_period[0]=drn4_pdata
    drn4 = flopy.mf6.ModflowGwfdrn(gwf,boundnames=True,save_flows=True, maxbound=len(drn4_pdata),\
                                stress_period_data=drn4_period,pname='drn4',
                                filename="{}_4.drn".format(model_name),)
    # Setup obs arrays for drn
    obs6_recarray = {
        "drn_channel_obs.csv": [
            ("drn4", "DRN", "channel")]
    }
    drn4.obs.initialize(
        filename="{}_4.drn.obs".format(model_name),
        digits=10,
        print_input=True,
        continuous=obs6_recarray,
    )
    
    mg=gwf.modelgrid
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    pmv = flopy.plot.PlotMapView(modelgrid=mg)
    pmv.plot_grid(ax=ax, lw=0.3, color="grey", alpha=0.3)
    pmv.plot_bc(name='channel',package=drn4, color = 'red')
    ax.set_title('Drain Cells Ephemeral Channels')
    ax.set_xlabel('Eastings')
    ax.set_ylabel('Northings')
    ax.ticklabel_format(style='plain') #  gets rid of the exponent offsets on the axis
    plt.tight_layout()
    
    spit = os.path.join(gis_f,"spit_outer_poly.shp")
    spit_nodes = get_bnodes(spit)
    
    drn5_pdata = [((lay,node),mg.top[node],0.001,"spit") for node in spit_nodes]
    drn5_period={}
    drn5_period[0]=drn5_pdata
    drn5 = flopy.mf6.ModflowGwfdrn(gwf,boundnames=True,save_flows=True, maxbound=len(drn5_pdata),\
                                stress_period_data=drn5_period,pname='drn5',
                                filename="{}_5.drn".format(model_name),)
    # Setup obs arrays for drn
    obs7_recarray = {
        "drn_spit_obs.csv": [
            ("drn5", "DRN", "spit")]
    }
    drn5.obs.initialize(
        filename="{}_5.drn.obs".format(model_name),
        digits=10,
        print_input=True,
        continuous=obs7_recarray,
    )
    
    mg=gwf.modelgrid
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    pmv = flopy.plot.PlotMapView(modelgrid=mg)
    pmv.plot_grid(ax=ax, lw=0.3, color="grey", alpha=0.3)
    pmv.plot_bc(name='spit',package=drn5, color = 'red')
    ax.set_title('Drain Cells South Pit')
    ax.set_xlabel('Eastings')
    ax.set_ylabel('Northings')
    ax.ticklabel_format(style='plain') #  gets rid of the exponent offsets on the axis
    plt.tight_layout()
    # annual rainfall mean = 1212 mm
    # we will start with 1% of annual rainfall
    # estimate of 0.5% to 2% of annual
    rain_rate = 1220/365/1000 # (m/d)
    min_rate = 0.005*rain_rate
    max_rate = 0.03*rain_rate
    # our preferred value which is central to log of min and max
    pv =  10**(np.log10(min_rate)+((np.log10(max_rate)-np.log10(min_rate))/2)) 

    # we also want to enhance recharge after the tailings deposition stops which is in stress period 194 (zero base)
    # so we ned an array of multipliers 1.33 increase recharge by 1/3 in the south pit only
    rch_mult_array=np.ones_like(mg.top)
    rch_mult_array[spit_nodes.astype('int')]=1.33


    rch_array_0=np.ones_like(mg.top)*pv
    rch_period = {}
    rch_period[0]=rch_array_0
    aux_period = {}
    aux_period[0]=[np.ones_like(mg.top)] # note these have to be lists with a number of arrays

    rch = flopy.mf6.ModflowGwfrcha(
        gwf,
        filename="{}.rch".format(model_name),
        pname="rch",
        fixed_cell=True,
        save_flows=True,
        recharge=rch_period,
        auxiliary='pit',
        auxmultname='pit',
        aux=aux_period
    )    
    pet = 1600 # mm/yr
    max_rate = pet/365/1000
    # assume our min rate is 50% less
    min_rate = 0.5*max_rate
    # our preferred value which is central to log of min and max
    pv =  10**(np.log10(min_rate)+((np.log10(max_rate)-np.log10(min_rate))/2)) 

    ext_depth = 1.0 #meters
    et_rate_array = np.ones_like(mg.top)*pv
    et_depth_array = np.ones_like(mg.top)*ext_depth
    et_period = {}
    evt = flopy.mf6.ModflowGwfevta(
        gwf,
        readasarrays=True,
        fixed_cell=False,
        surface = mg.top,
        rate = et_rate_array,
        depth = et_depth_array,
        filename="{}.evt".format(model_name),
        pname="evt")    
    # using hdata from before
    ihd_array=np.ones_like(mg.botm)
    ihd_array[:] = mg.top-5

    ic = flopy.mf6.ModflowGwfic(
        gwf, pname="ic", strt=ihd_array, filename="{}.ic".format(model_name)
    )    
    # building the output record for the head saving

    test = list(range(1,numper)) # this range represents the monthly stress periods before recovery
    hs_keys = [0,*test] # SSkey = 0, then all monthly SP

    h_rec = {key:[("HEAD","LAST")] for key in hs_keys}

    # combined head plus budget for zonebudget run
    zbud_rec = {key:[("BUDGET","LAST"),("HEAD","LAST")] for key in hs_keys}

    #for budget printing to list file
    b_rec = {key:[("BUDGET","LAST")] for key in hs_keys}

    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        budget_filerecord="{}.cbb".format(model_name),
        head_filerecord="{}.hds".format(model_name),
        headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=zbud_rec,
        printrecord=b_rec,
    )
    sim.write_simulation()
    return(sim, gwf)