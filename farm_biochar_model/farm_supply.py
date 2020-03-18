"""farm_supply: a library to handle energy equipment on the farm


"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import itertools

import farm_demand

def parsePLANTS_ficus(folder, file):
    """Parse an Excel dataframe and extracts a few key objects
    Usage: process, pro_comm, pro_class = fs.parsePLANTS(folder, file) 
    process = df with list of available processes, installed capacity, partload
    pro_comm = input output of processes
    pro_class = group processes by classes, define reference product of process
    pro_exc = mutually exclusive processes
    pro_bio = biosphere emission during use phase (e.g. air pollutants)
    """
    process = pd.read_excel(folder+file, sheet_name='Process', usecols='A:O')
    pro_comm = pd.read_excel(folder+file, sheet_name='Process-Commodity', usecols='A:E')
    pro_class = pd.read_excel(folder+file, sheet_name='Process-Class', usecols='A:F')
    pro_exc = pd.read_excel(folder+file, sheet_name='Process-Exclusive', usecols='A:E')
    pro_bio = pd.read_excel(folder+file, sheet_name='Process-Biosphere', usecols='A:F') # parsed, but not used yet
    
    return([process, pro_comm, pro_class, pro_exc, pro_bio])
 
def parsePLANTS(folder, file):
    """Parse an Excel dataframe and extracts a few key objects

    """
    plants = pd.read_excel(folder+file, sheet_name='process')
    commodities = pd.read_excel(folder+file, sheet_name='commodities')
    pro_com = pd.read_excel(folder+file, sheet_name='process-commodities')

    return([plants, commodities, pro_com])

    
    
def verifySizing(plants, commodity='heat', min_demand =0, max_demand=0):
    """Plots a diagram with min and max demand (kW) of a given commodity, usually heat, 
    and all the plant combinations. A warning is returned if there is gap in coverage.
    """
    ## Retrieve and calculate
    tmp = plants[(plants['type']=='production') & (plants['reference commodity']==commodity)]
    nb_plants =  len(tmp)
    x = [0, 1]
    combi = [p for p in itertools.product(x, repeat=3)]
    combi = np.array(combi)
    plant_set_cR = np.array([tmp['nominal capacity']*tmp['part load min'], tmp['nominal capacity']])
    capacities = combi.dot(np.transpose(plant_set_cR))
    
    ## Check gaps
        # visual inspection for now
        # print warning
    
    
    ## Plot
    fig, ax = plt.subplots(figsize=(15,4))
    i=0
    for row in itertools.product(x, repeat=3):
        ax.hlines(y=i+1, xmin=capacities[i][0], xmax=capacities[i][1], linewidth=2, color='r')
        i=i+1

    ax.vlines(x=min_demand, ymin=1, ymax=8, colors='b')
    if(max_demand == 0):
        max_demand = np.max(capacities)  
    ax.vlines(x=max_demand, ymin=1, ymax=8, colors='b')

    plt.text(capacities[2-1][1], 2, ' Electrical heater only', ha='left', va='center')
    plt.text(capacities[8-1][0], 8, 'All plants ', ha='right', va='center')

    plt.text(max_demand, 2, ' Max demand', ha='right', va='center')
    
    return(capacities)

def selectInterval(obj, start = '', end = ''):
    """Extract a subset from a smhi dataframe         
    """  
    select = obj[(obj.index >= start) & (obj.index <= end)]
    return(select)   
   
class energy_signature:
    def  __init__(self, area, heat_per_sqm_yr, water_per_year, area_ghg, elec_yr):
        self.area = area # area of buildings in sqm
        self.area_ghg = area_ghg # # area of greenhouse in sqm
        self.heat_per_sqm_yr = heat_per_sqm_yr # kWh/m2/yr
        self.heat_yearly = self.area * self.heat_per_sqm_yr #kWh/yr
        self.water_per_year = water_per_year # kWh/yr
        self.elec_yr = elec_yr # kWh/yr
    
    def get_profiles(self, filepath):
        profile = pd.read_csv(filepath, sep = ';', decimal = ',')
        return(profile)
    
    def get_energyDemand(self, filepath):
        profile = pd.read_csv(filepath, sep = ';', decimal = ',')
        profile = profile * list([1, self.heat_yearly, self.water_per_year, self.area_ghg, self.elec_yr])
        return(profile)

def run_scn_ficus_writer(plants_folder, plants_file, 
                        tb, tb_start, tb_end, calendar, 
                        commodities_imp, commodities_exp, import_max, export_max, 
                        el_folder, el_file, el_tab, ef_pellets, biochar_seq, 
                        plantsAvailable, farm_demand, year, elec_yr, 
                        afp, ficus_file, scenario):
                        
    print("Parsing sample_plants")
    process, pro_comm, pro_class, pro_exc, pro_bio = parsePLANTS_ficus(plants_folder,plants_file)

    print("Calculating data for ficus input folder")
    #Demand = collect from farm_demand
    final_commodities=['elec', 'heat']
    # calendar year = jan to decem; false, august to july year n+1
    monthday = ['-01-01','-12-31'] if calendar else ['-08-01','-07-31']
    year_txt = [year, year] if calendar else [year, str(int(year)+1)]
    
    tmp = selectInterval(farm_demand.resample('D').mean(), year_txt[0]+monthday[0], year_txt[1]+monthday[1])
    heat_to_df = tmp.sum(axis=1) #tmp['Heating_kW']+tmp['Water_kW']
    
    # tb_end : overwrite default value, by length of the year
    tb_end = len(list(heat_to_df))    
    if tb_end == 366:
        print("Annee bissextile")
    
    # Elec (quick fix)
    if calendar:
        folder = 'C:/Users/eazzi/Box Sync/KTH_PhD_IndustrialEcology/5_csLIN/data/'
        file = 'py_profileKimming.csv'
        hd_m = energy_signature(area=400, heat_per_sqm_yr=140, water_per_year=3361, area_ghg=100, elec_yr=elec_yr).get_energyDemand(folder+file)
        hd_m['month'] = pd.to_datetime([year+'-01-01',year+'-02-01',year+'-03-01',year+'-04-01',year+'-05-01',year+'-06-01', year+'-07-01',year+'-08-01',year+'-09-01',year+'-10-01',year+'-11-01',year+'-12-01'])
        hd_m = hd_m.set_index('month')
        tmp = np.array([])
        for index, row in hd_m.iterrows():
            if index.month in (1,3,5,7,8,10,12): #31 days
                tmp = np.append(tmp, np.repeat(row['electricity']/31/24,31))
            if index.month in (4,6,9,11): #30 days
                tmp = np.append(tmp, np.repeat(row['electricity']/30/24, 30))
            if index.month in (2,2): #28 or 29 days
                tmp = np.append(tmp, np.repeat(row['electricity']/28/24,28)) if tb_end == 365 else np.append(tmp, np.repeat(row['electricity']/29/24,29))
        elec_to_df = tmp
    else:
        elec_to_df = np.zeros(tb_end)
        
    #Time-Settings   
    df1 = pd.DataFrame({'Info':['Time'],
                        'timebase':[tb], # timebase, in s, e.g. hourly = 3600; monthly = 3600*730 (h/month); daily = 3600*24
                        'start':[tb_start], #
                        'end':[tb_end]}) # e.g. daily, 365 = last day of the year
    #MIP-Equations
    df2 = pd.DataFrame({'Equations':['Storage In-Out','Partload','Min-Cap'],
                        'Active':['no', 'yes', 'no']})
    #Ext-Commodities
    commodities = commodities_imp + list(set(commodities_exp) - set(commodities_imp)) # all commodities, unique list

    df3 = pd.DataFrame({'Commodity':commodities,
                        'demande-rate':list(np.repeat(0, len(commodities))),
                        'time-interval-demand-rate':list(np.repeat(df1['timebase'][0], len(commodities))),
                        'p-max-initial':list(np.repeat(0, len(commodities))),
                        'import-max':import_max,
                        'export-max':export_max,
                        'operating-hours-min':list(np.repeat(0, len(commodities))) })

    #Ext-Import = emission factor, at given time-step, for imported commodities
    #read Excel array with electricity data
    
    # IF CALENDAR YEAR
    ficus_powerEF = pd.ExcelFile(el_folder+el_file).parse(el_tab,index_col=[0],usecols='A,AE').fillna(0)
    ficus_powerEF = ficus_powerEF.resample('D').mean() # gCO2/kWh
    
    if not calendar:
        # Simulation starts on 1st of august, till 31 of july
        yy = np.append(ficus_powerEF.iloc[212:365], ficus_powerEF.iloc[0:212])
        
        if tb_end == 366: # add 29th of February to year+1
            yy = np.insert(yy, len(ficus_powerEF.iloc[212:365])+59, #append at that position
                            0.5*(yy[len(ficus_powerEF.iloc[212:365])+58]+yy[len(ficus_powerEF.iloc[212:365])+59])) # this value: average of 28/02 and 1/03  
    if calendar:
        #Simulation starts on 1st of January
        yy = np.array(ficus_powerEF)
        
        if tb_end == 366: # add 29th of February
            yy = np.insert(ficus_powerEF, 59,
                                        0.5*(ficus_powerEF.iloc[58]+ficus_powerEF.iloc[59])) # this value: average of 28/02 and 1/03  

    df4 = pd.DataFrame({
        'Time':list(np.linspace(1,df1['end'][0],df1['end'][0],dtype ='int')),
        'elec':list(np.squeeze(yy)), # replace by vector from Asterios
        'pellets': list(np.repeat(ef_pellets, df1['end'][0]))
    })

    #Ext-Export = emission factor, at given time-step, for exported commodities
    df5  = pd.DataFrame({
        'Time':list(np.linspace(1,df1['end'][0],df1['end'][0],dtype ='int')),
        'elec':list(np.squeeze(yy))
    })
    if 'biochar' in commodities:
        df5['biochar'] = list(np.repeat(biochar_seq, df1['end'][0])) # add only if biochar is in list of commodities


    #Demand-Rate-Factor = artefact from economic optim, set to 1 for all timesteps
    # replace by self-writing dictionnary
    dic = {'Time':list(np.linspace(1,df1['end'][0],df1['end'][0],dtype ='int'))}
    dic.update({x:list(np.repeat(1, df1['end'][0])) for x in commodities_imp})
    df6 =  pd.DataFrame(dic)

    #Process = collect from farm_supply
    df7 = process.loc[process['Process'].isin(plantsAvailable)].copy() #pd.DataFrame()
    
    #Process-Exclusive = collect from farm_supply
    df13 = pro_exc.loc[(pro_exc['Process1'].isin(plantsAvailable) & pro_exc['Process2'].isin(plantsAvailable))].copy() 
    
    #Process-Commodity = collect from farm_supply
    df8 = pro_comm.loc[pro_comm['Process'].isin(plantsAvailable)].copy() #pd.DataFrame()

    #Process-Class = define the reference flow of the process, for which commodity-ratios are defined
    df9 = pd.DataFrame({
        'Class':['ht', 'PV'],
        'Commodity':['heat','elec'],
        'Direction':['Out','Out'],
        'fee':[0,0],
        'cap-max':['inf','inf'],
        'energy-max':['inf','inf']        
    })

    #Storage = no storage term in the cases studied, just daily buffer
    df10 = pd.DataFrame({
        'Storage':[],
        'Commodity':[],
        'Num':[],
        'cost-inv-p':[],
        'cost-inv-e':[],
        'cost-fix-p':[],
        'cost-fix-e':[],
        'cost-var':[],
        'cap-installed-p':[],
        'cap-new-min-p':[],
        'cap-new-max-p':[],
        'cap-installed-e':[],
        'cap-new-min-e':[],
        'cap-new-max-e':[],
        'max-p-e-ratio':[],
        'eff-in':[],
        'eff-out':[],
        'self-discharge':[],
        'cycles-max':[],
        'lifetime':[],
        'DOD':[],
        'initial-soc':[],
        'depreciation':[],
        'wacc':[]
    })
    
    # read values from farm_demand
    df11 = pd.DataFrame({
        'Time': list(np.linspace(1,df1['end'][0],df1['end'][0],dtype ='int')),
        'elec': list(elec_to_df), 
        'heat': list(heat_to_df) 
    })

    #Suplm = e.g. solar for PV system
    df12 = pd.DataFrame({
        'Time':list(np.linspace(1,df1['end'][0],df1['end'][0],dtype ='int')),
        'solar':list(np.repeat(0, df1['end'][0]))
    })
    
    
    print("Writing ficus input file")

    ficus_scenario = scenario+'.xlsx'   
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    from openpyxl import load_workbook
    book = load_workbook(afp+'/farm_biochar_model/farm_ficus_input_template.xlsx') # in farm_biochar_model folder
    writer = pd.ExcelWriter(afp+ficus_file+ficus_scenario, engine='openpyxl') 
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    # Position the dataframes in the worksheet.
    df1.to_excel(writer, sheet_name='Time-Settings', startcol=1-1, startrow=2-1, header=False, index=False)
    df2.to_excel(writer, sheet_name='MIP-Equations', startcol=1-1, startrow=2-1, header=False, index=False)
    df3.to_excel(writer, sheet_name='Ext-Commodities', startcol=1-1, startrow=2-1, header=False, index=False)
    df4.to_excel(writer, sheet_name='Ext-Import', startcol=1-1, startrow=2-2, header=True, index=False)
    df5.to_excel(writer, sheet_name='Ext-Export', startcol=1-1, startrow=2-2, header=True, index=False)
    df6.to_excel(writer, sheet_name='Demand-Rate-Factor', startcol=1-1, startrow=2-2, header=True, index=False)
    df7.to_excel(writer, sheet_name='Process', startcol=1-1, startrow=2-1, header=False, index=False)
    df8.to_excel(writer, sheet_name='Process-Commodity', startcol=1-1, startrow=2-1, header=False, index=False)
    df9.to_excel(writer, sheet_name='Process-Class', startcol=1-1, startrow=2-1, header=False, index=False)
    df10.to_excel(writer, sheet_name='Storage', startcol=1-1, startrow=2-1, header=False, index=False)
    df11.to_excel(writer, sheet_name='Demand', startcol=1-1, startrow=2-2, header=True, index=False)
    df12.to_excel(writer, sheet_name='SupIm', startcol=1-1, startrow=2-2, header=True, index=False)
    df13.to_excel(writer, sheet_name='Process-Exclusive', startcol=1-1, startrow=2-2, header=True, index=False)
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    # Check file: overwrite with shorter df => old rows remain; fix that (do I need the copy, if I re-write all)? test it
    # add header true 