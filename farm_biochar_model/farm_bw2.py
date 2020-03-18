"""farm_bw2: a library to analyse the output of the model and parse a bw2 activity


"""
import pandas as pd
import numpy as np
import datetime

from brightway2 import *
from bw2temporalis import *
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import os
import csv

def get_EF(file, year):
    '''Gets EFs from input file, for import and export commodities. Returns a df in format for CC calculations
    EF_out have a negative sign (exports, e.g. biochar or surplus electricity)
    Usage:
    EF = get_EF(file)
    '''
    
    EF_in= pd.ExcelFile(file).parse('Ext-Import',index_col=[0]).fillna(0)
    dt = datetime.datetime(int(year),1,1)
    EF_in['datetime']= [dt+datetime.timedelta(days=x) for x in range(len(EF_in))]
    EF_in.set_index('datetime', inplace=True)

    EF_out = -1*pd.ExcelFile(file).parse('Ext-Export',index_col=[0]).fillna(0)
    EF_out['datetime']= [dt+datetime.timedelta(days=x) for x in range(len(EF_in))]
    EF_out.set_index('datetime', inplace=True)

    EF = pd.concat([EF_in, EF_out], axis = 1, keys=(['pro_p_in','pro_p_out']))
    EF.columns.rename(['direction','commodity'], inplace=True)

    return EF

processes = ['pyr','el. heater']

def calc_cc(processes, EF_file, r_process, r_demand, year, ficus_file, scenario, logfile):
    '''Calculates Inventory and Emissions for each timestep, all processes and final demand, and returns two dataframe lci_t and lcia_t
    Parameters:
    processes: list of processes e.g. ['pyr','el. heater'] (excluding final electricity consumption)
    EF_file: path to Excel file containing timed emission factors
    
    Usage:
    lci_t, lcia_t = calc_cc()
    '''
    
    EF = get_EF(EF_file, year)
    
    dflist_in = []
    dflist_ia = []
    p2e = 24
    dt = datetime.datetime(int(year),1,1)
    # grid not a process! remove it for loop excutions
    if 'grid' in processes:
        processes.remove('grid')

    for process in processes:
        #print(process)
        # get input output for process
        inout = r_process.loc[process].pivot_table(index=['t0'],columns=['commodity'], values=['pro_p_in', 'pro_p_out'])
        inout['datetime']= [dt+datetime.timedelta(days=x) for x in range(len(inout))]
        inout.set_index('datetime', inplace=True)
        inout.columns.rename(['direction','commodity'], inplace=True)  

        # log it loads
        heat_load = inout[inout['pro_p_out','heat']!=0]['pro_p_out','heat']
        
        writeLog([ficus_file+scenario+'_'+year, 'Load_average_'+process, np.mean(heat_load), 'kW heat output, yearly'],logfile)
        writeLog([ficus_file+scenario+'_'+year, 'Load_min_'+process, np.amin(heat_load), 'kW heat output, yearly'],logfile)
        writeLog([ficus_file+scenario+'_'+year, 'Load_max_'+process, np.amax(heat_load), 'kW heat output, yearly'],logfile)
        
        # log it time in use, for each process
        use1 = np.sum(inout['pro_p_out','heat']!=0)/365*100 # % annual
        writeLog([ficus_file+scenario+'_'+year, 'TimeInUse_year_'+process, use1, '% year'],logfile)
        
        use2 = use1*365/100/30.5 # nb months
        writeLog([ficus_file+scenario+'_'+year, 'TimeInUse_month_'+process, use2, 'months'],logfile)
        
        # multiply inout by cost matrix
        impact = inout.mul(EF*p2e, fill_value=0)#.sum(axis = 1, skipna = True)
        
        # add to dflist
        dflist_ia.append(impact)
        dflist_in.append(inout*p2e)
        
    # add final electrical demand
    cc_eld =EF['pro_p_in'].mul(r_demand*p2e, fill_value=0)
    cc_eld.columns.rename('commodity', inplace=True) 
    cc_eld2 = pd.concat([cc_eld], axis = 1, keys=(['pro_p_in']))
    cc_eld2.columns.rename(['direction','commodity'], inplace=True) 
    
    # eld =  pd.concat([pd.concat([r_demand.drop(['heat'], axis=1)], axis = 1, keys=(['pro_p_in']))], axis=1, keys=(['grid']))
    eld = pd.concat([r_demand.drop(['heat'], axis=1)], axis = 1, keys=(['pro_p_in']))*p2e
    eld.columns.rename(['direction','commodity'], inplace=True)

    dflist_ia.append(cc_eld2)
    dflist_in.append(eld)
    lcia_t = pd.concat(dflist_ia, axis = 1, keys=(processes))
    lcia_t.columns.rename(['process','direction','commodity'], inplace=True)
    
    lci_t = pd.concat(dflist_in, axis = 1, keys=(processes))
    lci_t.columns.rename(['process','direction','commodity'], inplace=True)
    
    # remove columns that have only zeros
    lcia_t = lcia_t.loc[:, (lcia_t != 0).any(axis=0)]
    lci_t = lci_t.loc[:, (lci_t != 0).any(axis=0)]
    
    ## set values near below 1e-3 to zero
    lcia_t[lci_t < 1e-3]=0
    lci_t[lci_t < 1e-5]=0
    
    return lci_t, lcia_t

def plot_cc(ficus_result_folder, lcia_t, sample='D'):
    '''Plot for LCA_t with signs
    pro_p_out > negative sign
    '''
    tmp = lcia_t.resample(sample).sum()/1000
    ax = tmp.plot.bar(stacked=True, width=1.0, figsize=(15,10))
    ax.set_ylabel("kg CO2-eq/"+sample)
    ax.figure.savefig(ficus_result_folder + '\\'+'lcia-'+sample+'-timeseries.png')  


def parse_MixData(filepath):
    MD = pd.ExcelFile(folder+file).parse('powermix-se',index_col=[0],usecols='A:S').fillna(0)
    return MD    
    
def get_apparentMix(process, lci_t, MD):
    '''Calculates annual apparent mix consumed, for each process at the farm
    It is based on daily consumption at the farm (lci_t) and grid mixes for Sweden, provided externally (M). M can be attributional or consequential
    Usage:
    A = get_apparentMix('grid', lci_t)
    Returns a DataFrame with ecoinvent keys (given in Excel sheet) name as columns
    '''
    # check that the key exist
    if (process,'pro_p_in','elec') in lci_t.columns:
        # c = vector, daily electric consumption on farm, for given process
        c = lci_t[process,'pro_p_in','elec']
        # M = technology mix in Sweden at given day
        #folder = 'C:/Users/eazzi/Box Sync/KTH_PhD_HeavyData/P2a_farm_biochar'+'/farm_bw2/data/'
        #file='powermix-se.xlsx'
        #MD = pd.ExcelFile(folder+file).parse('powermix-se_tweak_app',index_col=[0],usecols='A:L').fillna(0)
          # prod = pd.ExcelFile(folder+file).parse('powermix-se_tweak',index_col=[0],usecols='A:G').fillna(0)
          #  imp = pd.ExcelFile(folder+file).parse('powermix-se_tweak',index_col=[0],usecols='H:L').fillna(0)
          #  exp = pd.ExcelFile(folder+file).parse('powermix-se_tweak',index_col=[0],usecols='M').fillna(0)
                

        # e = summation vector
        e = np.ones(len(MD))

        # A = apparent annual mix consumed = (1/sum(C))*e.diag(C).M 
        A = pd.DataFrame(columns=MD.columns)
        A.loc['AppMix'] = (1/c.sum())*e.dot(np.diag(c).dot(MD))
        return A
    else:
        raise Exception('The process "{}" does not have electricity as input'.format(process))

def get_ecoinventMix(AppMix=''):
    '''converts an apparent mix from Swedish description to ecoinvent processes, based on correspondance matrix supplied
    Usage:
    eiA = get_apparentMix(AppMix)
    Returns a DataFrame with ecoinvent keys (given in Excel sheet) name as columns
    '''
    folder = 'C:/Users/eazzi/Box Sync/KTH_PhD_HeavyData/P2a_farm_biochar'+'/farm_bw2/data/'
    file='powermix-se.xlsx'
    MD = pd.ExcelFile(folder+file).parse('correspondance-table', index_col=[0],usecols='A:L').fillna(0)
    
    return MD   
    
# average elec input
def add_elecExchange(toActivity, process, lci_t, MD, EI, fg_db, scn):
    elec_in = lci_t[process,'pro_p_in','elec'].sum()/lci_t[process,'pro_p_out','heat'].sum() if process != 'grid' else 1
    # np.nanmean(lci_t[process,'pro_p_in','elec']/lci_t[process,'pro_p_out','heat'])
    # Add exchange with foreground process, to unit process of plant
    exchange = {
        'amount': elec_in, # kWh_el per kWh output process
        'unit': '1', #reference product
        'input': (fg_db, process+'_processelec_'+scn),
        'type': 'technosphere'
    }      
    toActivity['exchanges'].append(exchange)
    
    # Define 'process electricity' unit process, for each plant
    activity_pr = {
    'name': process+'_processelec_'+scn,
    'comment':'electricity consumed by plant, average annual mix perceived',
    'unit': 'kWh',
    'type': 'process',
    'exchanges': [{ 'output': (fg_db, process+'_processelec_'+scn), #self tuple
                    'input': (fg_db, process+'_processelec_'+scn), #self tuple
                    'type':'production',
                    'amount':1
                    }          
        ], 
    'tag_OM': 'O', # O = Operation = Use phase; M = Maintenance = installation, maintenance and disposal
    'tag': 'Process electricity' if process != 'el. heater' else 'Electrical heating', # Detailed tag
    }    
   
    print('## bw2 Calculate apparent mix for ... '+process)    
    A = get_apparentMix(process, lci_t, MD)
    # Convert to ecoinvent processes
    #EI = get_ecoinventMix()
    AEI = A.dot(EI.transpose())
    # Drop zeros
    AEI = AEI.loc[:, (AEI != 0).any(axis=0)]
    
    for tech in AEI.columns:
        # add exchange
        exchange = {
        'amount': AEI[tech]['AppMix'],
        'unit': 'kWh', #reference product
        'comment':'electricity mix',
        'input': eval(tech),
        'type': 'technosphere'
        }
        activity_pr['exchanges'].append(exchange) 
    
    return activity_pr


def writeLog(data, logfile):
    ''' Writes data to a csv logfile, for every run, called by run_scn_bw2
    row format: [scenario_name, feature, value, unit], the function adds a timestamp for when the log was made
    '''
    now = datetime.datetime.now().strftime('%Y%m%dT%H%M')
    data.append(now)
    with open(logfile, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)
 
def run_scn_bw2(ficus_result_folder, ficus_result_file, year, calendar, p2e, plantsAvailable,ficus_folder, ficus_file, scenario, proj, fg_db, fresh_db=False, plantEmissions=False, plantManufacturing=False, show_plot=False):
    '''Generates an LCA-graph and saves it to bw2 project, also saves some graphs. Main input is the result file generated by farm_ficus. 
    Also here that background (ecoinvent) processes are selected and that tags (for graph traversal are defined)
    Returns lci_t, lcia_t
    Function is called in the workflow:
        lci_t, lcia_t = fb.run_scn_bw2(result_dir, 'result-'+proj+'_'+scenario+'.xlsx', year, p2e, 
                      plantsAvailable, ficus_folder, ficus_file, scenario,
                      proj, fg_db, fresh_db=False, plantEmissions=False, plantManufacturing=False)   
    '''
    
    logfile = ficus_folder+ficus_file+'filelog.csv'
    
    folder = ficus_result_folder
    file = ficus_result_file
    resultfile=folder+'/'+file
    # Parse
    r_demand = pd.ExcelFile(resultfile).parse('Demand timeseries',index_col=[0]).fillna(0)
    r_process = pd.ExcelFile(resultfile).parse('Process timeseries',index_col=[0]).fillna(0)
    r_external = pd.ExcelFile(resultfile).parse('External timeseries',index_col=[0]).fillna(0)
    # Set index, date
    dt = datetime.datetime(int(year),1,1)
    r_demand['datetime']= [dt+datetime.timedelta(days=x) for x in range(len(r_demand))]
    r_demand.set_index('datetime', inplace=True)
    
    #######################
    # Biochar production potential: log, plot, save
    if 'biochar' in r_external.index:
        # a biochar scenario   
        biochar_production = r_external.loc['biochar'][['t0','ext_p_out']]
        biochar_production.rename(columns={'ext_p_out':'biochar_production'}, inplace=True)
        biochar_production.set_index('t0', inplace=True)
        biochar_production=biochar_production*p2e
        print("Biochar produced annually: ", biochar_production.sum()/1000, " tons")
        # log it
        writeLog([ficus_file+scenario+'_'+year, 'biochar production', biochar_production.values.sum()/1000, 't/y'],
                logfile)
        
        # plot it and save
        ax = biochar_production.plot(figsize=(15,10))
        biochar_production.cumsum().plot(ax = ax, secondary_y=True)
        ax.figure.savefig(ficus_result_folder + '\\'+'biochar'+'-timeseries.png')
        plt.close(ax.figure)
        
        # plot distribution
        distrib = np.array(biochar_production.sort_values(by=['biochar_production'], ascending=False))
        distrib = np.trim_zeros(distrib, 'b') # Trim the last zeros
        distrib = np.append(distrib, 0) # Add one zero to scale the figure
        
        fig, ax = plt.subplots(figsize=(10,10)) 
        plt.plot(distrib, 'b-')
        plt.xlabel("Days", fontsize=16)
        plt.ylabel("Biochar production load, kg/day", fontsize=16)
        ax.grid(True)
        plt.savefig(ficus_result_folder + '\\'+'biochar'+'-LDC.png', dpi=300)        
        plt.close(ax.figure)
    #######################
        
    
    
    #######################
    print("Compile ficus lci and lcia")
    lci_t = pd.DataFrame({})
    lcia_t = pd.DataFrame({})
    lci_t, lcia_t = calc_cc(plantsAvailable, ficus_folder+ficus_file+scenario+'.xlsx', r_process, r_demand, year, ficus_file, scenario, logfile)
    print(lcia_t.resample('Y').sum().sum().sum()/1e6, "ton CO2-eq, net score, for 1 year of farm-energy")
    
    if show_plot:
        plot_cc(ficus_result_folder, lcia_t, 'd')
        plot_cc(ficus_result_folder, lcia_t, 'w')
    
    
    #######################
    print("Towards bw2...")
    # This retirieves your Windows username
    user=os.getenv('USERNAME')
    #This sets where you want the folder of your project to be.
    os.environ['BRIGHTWAY2_DIR'] = "C:\\Users\\"+user+"\\Box Sync\\BrightwayAppData\\"
    # Check if project exists
    if proj not in projects:
        projects.create_project(proj)
    projects.set_current(proj)
    # Set-up project if not already
    if "biosphere3" not in databases:
        bw2setup()
    # Set up ecoinvent if not already
    ei_name = "ecoinvent 3.5 cutoff"
    ei_path = "C:\\Users\\eazzi\\Box Sync\\KTH_PhD_HeavyData\\ecoinvent_3.5_cutoff_ecoSpold02\\datasets\\"
    if ei_name not in databases:
        # add ecoinvent to project
        ei = SingleOutputEcospold2Importer(ei_path, ei_name)
        ei.apply_strategies()
        ei.write_database()
    # to make the calculation faster we set ecoinvent to static as explained in the documentation
    # databases[ei_name]['static'] = True
    # databases.flush()
    # Call db
    eidb = Database('ecoinvent 3.5 cutoff')
    CO2 = ('biosphere3', 'f9749677-9c9f-4678-ab55-c607dfdc2cb9') 
    # Biomass supply chains
    tpl_pellets = ('ecoinvent 3.5 cutoff', 'c4689b656b80f2e2145097471ce7a789') # TUPLE market for wood pellet	RER	kilogram
    tpl_woodchip = ('ecoinvent 3.5 cutoff', 'f29f12860f4ff52602e14897c0b82184') # TUPLE softwood forestry, spruce, sustainable forest management, SE
    tpl_willowSLU = ('farmEnergySystem', '2d0325d83cb642a28eecea625c2e8f6c') # TUPLE based on own process
    tpl_willowEI = ('ecoinvent 3.5 cutoff', '997439425b362b8eb9eba237151c3569') # TUPLE wood chips and particles, willow, DE
    tpl_wasteAvoidedDecay = ('farmEnergySystem', 'e0196f83a38141509c9971fb97c7b453_copy1') # TUPLE based on own process

    # Plant manufacturing
    tpl_plant = ('ecoinvent 3.5 cutoff', 'e16c97aa0596bc2ed3f5472fec0e43a9') # TUPLE furnace production, pellet, 50kW	CH	unit # used for combustion and pyrolysis plant
    tpl_hp = ('ecoinvent 3.5 cutoff', 'bc1ed1c554e8101a767ef6e505227a0b') # TUPLE name: heat pump production, brine-water, 10kW, CH; lifetime: 20 years
    
    tpl_borehole = ('ecoinvent 3.5 cutoff', 'e65ea9034268ee4d2842baa289f65b7e') # TUPLE name: borehole heat exchanger production, 150m CH for 10 kW hp; lifetime 50 years
    tpl_gshp = ('ecoinvent 3.5 cutoff', 'bc1ed1c554e8101a767ef6e505227a0b') # TUPLE name: heat pump production, brine-water, 10kW, CH; lifetime: 20 years

    manuf_plantsAvailable = { # 0: manufacturing process / 1: plant lifetime / 2: biomass supply / 3: biochar C content / 
                              # 4: biochar stability / 5: LHV_dry_pellets
        'hp': [tpl_hp, 20, tpl_pellets, 0, 0, 0],
        'gshp30':[tpl_gshp, 20, tpl_pellets, 0, 0, 0],
        'pyr': [tpl_plant, 20, tpl_pellets, .77, .89, 18.35],
        'pyr-el':[tpl_plant, 20, tpl_pellets],
        'el. heater':(),
        'pv':(),
        'comb':[tpl_plant, 20, tpl_pellets, 0, 0, 18.35],
        'pyrBMC60':[tpl_plant, 20, tpl_pellets, .77, .89, 18.35],
        'BioGreen60':[tpl_plant, 20, tpl_pellets, .77, .89, 18.35],
        'pyrBMC160':[tpl_plant, 20, tpl_pellets, .77, .89, 18.35],
        'pyrBMC250':[tpl_plant, 20, tpl_pellets, .77, .89, 18.35],
        'BioGreen3700':[tpl_plant, 20, tpl_pellets, .77, .89, 18.35],
        'CleanFuels2800':[tpl_plant, 20, tpl_pellets, .77, .89, 18.35],
        'Pyreg1500':[tpl_plant, 20, tpl_pellets, .77, .89, 18.35],
        'Pyreg500':[tpl_plant, 20, tpl_pellets, .77, .89, 18.35],
        'pyrBMC60t':[tpl_plant, 20, tpl_pellets, .77, .89, 18.35],
        'pyrBMC60t_wh':[tpl_plant, 20, tpl_pellets, .77, .89, 18.35],
        'pyrBMC30t':[tpl_plant, 20, tpl_pellets, .77, .89, 18.35],
        'pyrBMC50t':[tpl_plant, 20, tpl_pellets, .77, .89, 18.35],
        'pyrBMC70t':[tpl_plant, 20, tpl_pellets, .77, .89, 18.35],
        'pyrBMC50t_el':[tpl_plant, 20, tpl_pellets, .77, .89, 18.35], #el-boosted
        'pyrBMC50t_b1':[tpl_plant, 20, tpl_pellets, .77, .89, 18.35], #alt biomass: wood pellet
        'pyrBMC50t_b2':[tpl_plant, 20, tpl_woodchip, .77, .89, 17.89], #alt biomass: woodchips
        'pyrBMC50t_b3':[tpl_plant, 20, tpl_willowSLU, .77, .89, 17.89], #alt biomass: willow SLU
        'pyrBMC50t_b4':[tpl_plant, 20, tpl_willowEI, .77, .89, 17.89], #alt biomass: willow EI
        'pyrBMC50t_b5':[tpl_plant, 20, tpl_wasteAvoidedDecay, .65, .89, 17.0], #alt biomass: waste silage, avoided decay
        
    }
    # Parsing LCI data for Manufacturing, from Excel
    plants_folder= 'C:/Users/eazzi/Box Sync/KTH_PhD_HeavyData/P2a_farm_biochar'+'/farm_supply/'
    plants_file= 'sample_plants.xlsx'
    Pro_Manuf = pd.read_excel(plants_folder+plants_file, sheet_name='Process-Manufacturing')
    Pro_Manuf.set_index('Process', inplace=True)
        
    # FU definition
    scn=scenario # name of scenario
    activity_farmEnergySystem = {
        'name': 'farmEnergy_'+scn,
        'comment':'One year of heat and power supplied at the farm',
        'unit': '1',
        'type': 'process',
        'exchanges': [{ 'output': (fg_db, 'farmEnergy_'+scn), #self tuple
                        'input': (fg_db, 'farmEnergy_'+scn), #self tuple
                        'type':'production',
                        'amount':1
                    }]  
    }
    print("bw2# Add production processes, manufacturing, and start-ups")
    # Add production processes
    processes = plantsAvailable
    for process in processes:  
        if process == 'grid': #BUG: Never appears in the output; 'grid' must have been changed? grid never in plantsAvailable
            z = lci_t[process,'pro_p_in','elec'].sum()
            exchange = {
                'amount': z, # reference flow is elec
                'unit': 'kWh', #reference product
                'comment':'kWh of reference product of the plant',
                'input': (fg_db, process+'_'+scn),
                'tag': 'electricity_demand',
                'type': 'technosphere'
            }
            activity_farmEnergySystem['exchanges'].append(exchange)
            # log it
            writeLog([ficus_file+scenario+'_'+year, 'final electricity', z, 'kWh/y'],logfile)
        else:
            z = lci_t[process,'pro_p_out','heat'].sum()
            exchange = {
                'amount': z, # reference flow is output heat
                'unit': 'kWh', #reference product
                'comment':'kWh of reference product of the plant',
                'input': (fg_db, process+'_'+scn),
                'type': 'technosphere'
            }
            activity_farmEnergySystem['exchanges'].append(exchange)
            # log it
            writeLog([ficus_file+scenario+'_'+year, process+' heat produced', z, 'kWh/y'],logfile)
            
            # if not el. heater
            if not 'el. heater' in process:
                # add manufacturing
                val = 1/Pro_Manuf.loc[process]['lifetime'] if plantManufacturing else 0 # still write it, to allow easy edit in AB
                exchange = {
                    'amount': val, # reference flow is output heat
                    'unit': 'unit/year', #reference product
                    'input': (fg_db, process+'_manufacturing_'+scn),
                    'type': 'technosphere'
                }
                activity_farmEnergySystem['exchanges'].append(exchange)
            
                # add number of plant start-up and shut-down
                ss = lci_t[process,'pro_p_in','elec'].copy() # valid for all processes consuming electricity
                ss = np.diff(np.divide(ss, ss, out=np.zeros_like(ss), where=ss!=0))
                val = len(ss[ss>0]) #  len(ss[ss!=0]) = start up and shut down; len(ss[ss>0]) = only start-ups ; len(ss[ss<0]) = only shut-down
                exchange = {
                    'amount': val, # reference flow is output heat
                    'unit': 'unit/year', #reference product
                    'comment':'start-up',
                    'input': (fg_db, process+'_startup_'+scn), # define what is consumed at start-up; own activity! to define elsewhere
                    'type': 'technosphere'
                }
                activity_farmEnergySystem['exchanges'].append(exchange)
    
    # Save FU and production processes in dictionnary data
    if fresh_db==True:
        data = {} # Reset the database every time we run it!  
    else:
        db = Database(fg_db) # loads the data existing, the new scenario will be pushed to it
        data = db.load()
    
    data.update([ ( (fg_db,'farmEnergy_'+scn) ,activity_farmEnergySystem) ] )
    #pprint(data, width=140)    
    
    print("bw2# Add technosphere and biosphere flows to each production process")
    # Add inputs and outputs to each production process
    processes = lci_t.columns.levels[0]
    direction = lci_t.columns.levels[1]
    commodities = lci_t.columns.levels[2]
 
    # get EI
    EI = get_ecoinventMix() # correspondance table
    # get MD = technology mix in Sweden grid at given day
    folder = 'C:/Users/eazzi/Box Sync/KTH_PhD_HeavyData/P2a_farm_biochar'+'/farm_bw2/data/'
    file='powermix-se.xlsx'
    MD = pd.ExcelFile(folder+file).parse('powermix-se_tweak_app',index_col=[0],usecols='A:L').fillna(0)
    MD = MD.resample('D').mean()
    MD = MD.reset_index(drop=True) #  daily mix
    # row3: fictious 29th of February, as average of day before, day after; PROXY
    row1 = pd.DataFrame(MD.loc[58]).transpose()
    row2 = pd.DataFrame(MD.loc[59]).transpose()
    row3 = row2 
    row3.iloc[0] = 0.5*(row1.values+row2.values) # this mix: average of 28/02 and 1/03  
 
    if not calendar:
        # Simulation starts on 1st of august, till 31 of july
        MD = pd.concat([MD.iloc[212:365], MD.iloc[0:212]]).reset_index(drop=True)
                
        if len(r_demand) == 366: # add 29th of February to year+1
            MD = pd.concat([MD.iloc[:153+58], row3, MD.iloc[153+58:]]).reset_index(drop=True)

    if calendar:
        #Simulation starts on 1st of January
        if len(r_demand) == 366: # add 29th of February
            MD = pd.concat([MD.iloc[:58], row3, MD.iloc[58:]]).reset_index(drop=True) 
    
    # Get biosphere emissions for plantsAvailable
    plants_folder= 'C:/Users/eazzi/Box Sync/KTH_PhD_HeavyData/P2a_farm_biochar'+'/farm_supply/'
    plants_file= 'sample_plants.xlsx'
    pro_bio = pd.read_excel(plants_folder+plants_file, sheet_name='Process-Biosphere', usecols='A:F')
    pro_bio = pro_bio.pivot_table(index='Biosphere_key', columns=['Process'], values=['ratio'])
   
    # Define time-dependent unit processes: /kWh ref product
    for process in processes:
        activity_pr = {
        'name': process+'_'+scn,
        'comment':'On-farm process',
        'unit': 'kWh',
        'type': 'process',
        'exchanges': [{ 'output': (fg_db, process+'_'+scn), #self tuple
                        'input':  (fg_db, process+'_'+scn), #self tuple
                        'type':'production',
                        'amount':1
                    }] 
        }
        for commodity in commodities:
            if (process,'pro_p_in',commodity) in lci_t.columns:
                # commodity is an input to the process
                if commodity == 'elec':
                    # update database dictionary with current process
                    data.update([ ( (fg_db, process+'_processelec_'+scn), add_elecExchange(activity_pr, process, lci_t, MD, EI, fg_db, scn)) ]) 
                    # log it
                    writeLog([ficus_file+scenario+'_'+year, process+' elec input', lci_t[process,'pro_p_in','elec'].sum(), 'kWh/y'],logfile)
                    
                if commodity == 'pellets': # TODO: Need to change for future biomass comparison > commodity in solid_fuels = ['tree','ep']
                    # average pellet consumption per kWh heat produced
                    LHV_pellets = manuf_plantsAvailable[process][5]/3.6 #kWh LHV dry per kg 
                    pellet_in = lci_t[process,'pro_p_in','pellets'].sum()/lci_t[process,'pro_p_out','heat'].sum()/LHV_pellets # kg pellets / kWh heat
                    # np.nanmean(lci_t[process,'pro_p_in','pellets']/lci_t[process,'pro_p_out','heat'])/LHV_pellets
                    # add exchange
                    exchange = {
                    'amount': pellet_in, # average kg dry pellet per average 1kWh heat produced
                    'unit': 'kilogram', #reference product
                    'comment':'average kg dry pellet per average 1kWh heat produced',
                    'input': (fg_db, process+'_biomassfuel_'+scn),
                    'type': 'technosphere',
                    }
                    activity_pr['exchanges'].append(exchange)
                    # log it
                    writeLog([ficus_file+scenario+'_'+year, process+' pellet input', lci_t[process,'pro_p_in','pellets'].sum(), 'kWh/y'],logfile)
            
            if (process,'pro_p_out',commodity) in lci_t.columns:
                if commodity == 'biochar':
                    #Innacurate: biochar_out = np.nanmean(lci_t[process,'pro_p_out','biochar']/lci_t[process,'pro_p_out','heat']) # average ratio biochar out / heat out
                    biochar_out = lci_t[process,'pro_p_out','biochar'].sum()/lci_t[process,'pro_p_out','heat'].sum() # avoid nan issues when dividing 0, the accuracy loss, and confusion between means (with/without nans)
                    exchange = {
                        'negative': True,
                        'amount': -1*biochar_out*44/12*manuf_plantsAvailable[process][3]*manuf_plantsAvailable[process][4], # average kg CO2 sequestred per average 1kWh heat produced
                        'unit': 'kilogram', #reference product
                        'comment':'average kg CO2 sequestred per average 1kWh heat produced',
                        'input': CO2,
                        'tag_OM': 'O', # O = Operation = Use phase; M = Maintenance = installation, maintenance and disposal
                        'tag': 'Carbon sequestration',
                        'type': 'biosphere'
                    }
                    activity_pr['exchanges'].append(exchange)
                    
                    
        # add biosphere exchanges
        if process in pro_bio.columns.levels[1]:
            for key in pro_bio.index:
                val = pro_bio.loc[key]['ratio',process] if plantEmissions else 0 # still write it, to allow easy edit in AB
                exchange = {
                    'negative': False,
                    'amount': val, 
                    'unit': 'kilogram', #reference product
                    'comment':'plant emissions during use phase,  kg per kWh heat produced',
                    'input': eval(key),
                    'type': 'biosphere',
                    'tag_OM': 'O', # O = Operation = Use phase; M = Maintenance = installation, maintenance and disposal
                    'tag': 'Plant emissions'
                }
                activity_pr['exchanges'].append(exchange)
              

        # update database dictionary with current process
        data.update([ ( (fg_db, process+'_'+scn),
                          activity_pr) ])     

    
    for process in processes:
        if process not in ['el. heater', 'grid']:
            #it is either pyr or comb process
            
            # Define 'start-up' unit process, for each plant
            activity_pr = {
            'name': process+'_startup_'+scn,
            'comment':'one-time plant start-up',
            'unit': '1',
            'type': 'process',
            'exchanges': [{ 'output': (fg_db, process+'_startup_'+scn), #self tuple
                        'input':  (fg_db, process+'_startup_'+scn), #self tuple
                        'type':'production',
                        'amount':1
                    } #empty for now, can be edited in activity browser
                    ],
            'tag_OM': 'O', # O = Operation = Use phase; M = Maintenance = installation, maintenance and disposal
            'tag': 'Plant start-up', # Detailed tag               
            }
            # update database dictionary with current process
            data.update([ ( (fg_db, process+'_startup_'+scn), activity_pr) ])     

            # Define 'manufacturing' unit process, for each plant
            activity_pr = {
            'name': process+'_manufacturing_'+scn,
            'comment':'plant start-up',
            'unit': '1',
            'type': 'process',
            'exchanges': [{ 'output': (fg_db, process+'_manufacturing_'+scn), #self tuple
                        'input':  (fg_db, process+'_manufacturing_'+scn), #self tuple
                        'type':'production',
                        'amount':1}], 
            'tag_OM': 'M', # O = Operation = Use phase; M = Maintenance = installation, maintenance and disposal
            'tag': 'Manufacturing', # Detailed tag                
            }
            # Add manufacturing exchanges based on Excel file
            obj = eval(Pro_Manuf.loc[process]['Activity_Key'])
            if isinstance(obj, list):
                keys = eval(Pro_Manuf.loc[process]['Activity_Key'])
                amounts = eval(Pro_Manuf.loc[process]['Amount'])
                names = eval(Pro_Manuf.loc[process]['Activity_Name'])
                for n in range(len(obj)):
                    exchg = {
                            'amount': float(amounts[n]), # plant
                            'unit': 'unit', # plant
                            'comment':'manufacturing plant unit',
                            'input': keys[n],
                            'type': 'technosphere'
                                }
                    activity_pr['exchanges'].append(exchg)
            else:
                # single process
                exchg = {
                    'amount': float(Pro_Manuf.loc[process]['Amount']), # plant
                    'unit': 'unit', # plant
                    'comment':'manufacturing plant unit',
                    'input': eval(Pro_Manuf.loc[process]['Activity_Key']),
                    'type': 'technosphere'
                        }
                activity_pr['exchanges'].append(exchg)
            
            # update database dictionary with current process
            data.update([ ( (fg_db, process+'_manufacturing_'+scn), activity_pr) ])   

            # Define 'biomassfuel' unit process, for each plant
            activity_pr = {
            'name': process+'_biomassfuel_'+scn,
            'comment':'fuel consumed by plant',
            'unit': 'kilogram',
            'type': 'process',
            'exchanges': [{ 'output': (fg_db, process+'_biomassfuel_'+scn), #self tuple
                        'input':  (fg_db, process+'_biomassfuel_'+scn), #self tuple
                        'type':'production',
                        'amount':1
                        },
                        {
                        'amount': 1, # kg of biomass
                        'unit': 'kilogram', # plant
                        'comment':'biomass fuel',
                        'input': manuf_plantsAvailable[process][2],
                        'type': 'technosphere'
                            }], 
            'tag_OM': 'O', # O = Operation = Use phase; M = Maintenance = installation, maintenance and disposal
            'tag': 'Fuel production' # Detailed tag 
            }
            # update database dictionary with current process
            data.update([ ( (fg_db, process+'_biomassfuel_'+scn), activity_pr) ])  
                          
    print('Current project is: ',projects.current)
    print('Read only mode: ',projects.read_only)
    db = Database(fg_db)
    db.write(data)
    
    print('Scenario ', scn, ' is now ready for analysis in Activity-Browser')
    
    return lci_t, lcia_t
    
    

    
    
