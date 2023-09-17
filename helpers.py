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