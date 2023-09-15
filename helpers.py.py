import os
import sys
import flopy

def ws1_mod(sim):
    #ws = './model'
    name = 'MySim'
    #sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name='mf6')
    mod_name = 'MyModel'
    tdis = flopy.mf6.ModflowTdis(sim)
    print('building tdis package')
    ims = flopy.mf6.ModflowIms(sim)
    print('building ims package')
    gwf = flopy.mf6.ModflowGwf(sim, modelname=mod_name, save_flows=True)
    print('building ')
    dis = flopy.mf6.ModflowGwfdis(gwf, nrow=10, ncol=10)
    ic = flopy.mf6.ModflowGwfic(gwf)
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True)
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, 0), 1.],
                                                           [(0, 9, 9), 0.]])
    budget_file = mod_name + '.bud'
    head_file = mod_name + '.hds'
    oc = flopy.mf6.ModflowGwfoc(gwf,
                                budget_filerecord=budget_file,
                                head_filerecord=head_file,
                                saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
    return()