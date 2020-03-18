"""farm_demand: a library to compute the energy demand of a farm

It can either parse energy consumption data collected at the farm, model energy
consumption data from weather data, or use energy consumption profiles. 
The data can be aggregate or resampled from hourly to yearly data.
"""


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import itertools


class clientSMHI:
    """Client to parse list of xlsx files from SMHI (weather stations in Sweden)
    and return a dataframe with date-time as index

    Use:
    s = clientSMHI()
    smhi = s.parseSMHI(folder, files)

    """
    def __init__(self):
        """To do: get data from SMHI API, live, based on location (nearest weather station) and dates
        """
        pass

    def parseSMHI(self, folder, files):
        """Client to parse list of xlsx files from SMHI (weather stations in Sweden)
        and return a dataframe with date-time as index

        """
        smhi = pd.DataFrame()
        tmp = list()
        for file in files:
            print('Parsing SMHI weather data: '+file)
            tmp.append(pd.read_csv(folder+file, sep = '\s+', decimal = '.', parse_dates=True))
            smhi = pd.concat(tmp, ignore_index=True) #sort=True
            smhi['Year'] = pd.to_datetime(smhi['Datum']).dt.year
            smhi['Month'] = pd.to_datetime(smhi['Datum']).dt.month
            smhi['Day'] = pd.to_datetime(smhi['Datum']).dt.day
            smhi['Datetime'] = pd.to_datetime(smhi['Datum'] + ' ' + smhi['Tid_(UTC)'])
            smhi = smhi.set_index('Datetime')
            # smhi['Julian'] = pd.to_datetime(smhi['Datum']).dt.day
        return(smhi)

    def saveSMHI(self, smhi, folder, filename):
        """Save the smhi dataframe to a local file         
        """  
        pass

    def loadSMHI(self, folder, filename):
        """Load the smhi dataframe from a local file       
        """  
        pass

def selectInterval(obj, start = '', end = ''):
    """Extract a subset from a smhi dataframe
        start / end given as string, 2018-01-01
    """  
    select = obj[(obj.index >= start) & (obj.index <= end)]
    return(select)
     
def HeatingSignature(smhi, ref_year, tot_nrj, T_max):
    """Calculate the heating energy signature based on few input parameters:
    - smhi = temperature data
    - ref_year = string, a year used as reference e.g. '2017', must be in smhi data; or a list two years of years, e.g. ['1996', '2010']
    - tot_nrj = annual SPACE heating consumed, in kWh
    - T_max = temperature above which heating is turned off on the farm, and demand is considered null
    
    Linear relationship between kW_heating and outdoor temperature, defined as : kW_heating  = A*(T-T_max) and 0 if T>T_max
    - A in kW/K 
    Returns, A, T_max
    
    """
    if isinstance(ref_year, str):
        # ref_year is a string, we use a single year as reference
        start_yr = ref_year
        end_yr  = ref_year
        nb_yr = 1
    else:
        # a list should have been passed, with two arguments
        start_yr = ref_year[0]
        end_yr  = ref_year[1]
        nb_yr = int(ref_year[1])-int(ref_year[0])
        
    tp_year = selectInterval(smhi, start_yr+'-01-01', end_yr+'-12-31')
    DeltaT=tp_year['Lufttemperatur']-T_max
    DegreeDays = DeltaT[DeltaT < 0].sum()
    A = nb_yr*tot_nrj/DegreeDays
    
    print("Building heating signature: ", A, " kW/K") 
    print("Average yearly degreedays: ", DegreeDays/nb_yr, " degree-hour/year")
    return([A, T_max]) 
    
def smhi_ccf(row, cc_scen):
    '''SMHI Climate change forecast
    Simple prediction: +3 degrees on average temperature, per season
    Improvement option: SU, MISU, Evelien contact for CC forecast, air temperature timeseries
    
    '''
    if row['Month'] in (12,1, 2):
        return row['Lufttemperatur']+cc_scen[0] # winter increase
    if row['Month'] in (3,4,5):
        return row['Lufttemperatur']+cc_scen[1] # spring increase
    if row['Month'] in (6,7,8):
        return row['Lufttemperatur']+cc_scen[2] # summer increase
    if row['Month'] in (9,10,11):
        return row['Lufttemperatur']+cc_scen[3] # fall increase
    return row['Lufttemperatur']
    
def HeatingDemand(smhi, ref_year, tot_nrj, T_max, cc=False, cc_scen=[3,3,3,3]):
    """Calculate the heating energy signature based on few input parameters:
    - smhi = temperature data
    - ref_year = string, a year used as reference e.g. '2017', must be in smhi data; or a list of years, e.g. ['1996', '2010']
    - tot_nrj = annual heating consumed, in kWh
    - T_max = temperature above which heating is turned off on the farm, and demand is considered null
    
    - cc = simulate climate change effect in Sweden
    - cc_scen = vector with average temperature increase per season (order: Winter > Spring > Summer > Fall), according to SMHI; e.g. [3,3,3,3]
    
    Linear relationship between kW_heating and outdoor temperature, defined as : kW_heating  = A*(T-T_max) and 0 if T>T_max
    
    Returns a dataframe with HeatingDemand for the whole smhi data set
    
    """
    col = 'Lufttemperatur'
    new_col = 'Heating_kW'
    A, T_max = HeatingSignature(smhi, ref_year, tot_nrj = tot_nrj, T_max = T_max) # ref year unchanged, even with cc=True
    if cc:
        # add to smhi a new column, Lufttemperatur_cc 
        smhi['Lufttemperatur_cc'] = smhi.apply(lambda row: smhi_ccf(row, cc_scen), axis=1)
        # change column
        col = 'Lufttemperatur_cc'
        
    tmp = A*(smhi[col]-T_max)
    tmp[smhi[col]-T_max > 0] = 0
    tmp = pd.DataFrame(tmp)
    tmp.rename(columns={col:new_col}, inplace=True)
    return(tmp)

def BoverketToHeat(areas, energiprestande, water_share, cop_corr):
    '''Calculates, for a normal year, the total heating demand (space heating and hot water demand) 
    based on the properties of the buildings, as given in Swedish energy declarations (Boverket)
    Values are representative of a "normal year"
    
    Usage:
    tot_njr, tot_heat, tot_water = BoverketToHeat(areas, energiprestande, water_share, cop_corr)
    '''
    e_ht = np.multiply(energiprestande, cop_corr) # kWh_heat/m2/yr, convert Boverket values to demand values; /PEF *COP; worst casem el values
    tot_njr = areas.dot(np.transpose(e_ht))
    tot_water = areas.dot(np.multiply(e_ht, water_share))
    tot_heat = tot_njr - tot_water
    print('Total heat demand is ', tot_njr, ' kWh of which', tot_water/tot_njr*100, '% hot water')
    return tot_njr, tot_heat, tot_water
    
def SelectSamplePlot(df, values, start = '', end = '', sample=''):
    """Takes a dataframe (smhi, demand) dimension, selects a sub set, resample it and plot it
    
    """ 
    tmp = selectInterval(df, start, end)
    tmp = tmp.resample(sample).mean()
    table = pd.pivot_table(tmp, values=values, index=['Datetime'], columns=[])
    plt.figure(figsize=(18,9))
    table.plot()
    
def LDC(df, values, start = '', end = ''):
    """Returns the LDC of a given vector, with start and end dates
    """
    LDC = list(selectInterval(df, start, end)[values].sort_values(ascending=False))
    plt.figure(figsize=(18,9))
    LDC_plot = plt.plot(LDC)
    return(LDC)

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

        
def run_scn_demand(smhi_folder, smhi_files, areas, energiprestande, water_share, cop_corr, ref_year, T_max, cc, cc_scen):
    ''' Returns the heating demand estimates for space heating and hot water production based on energy declarations and historic temperature series, using the degree day methodology
        
        Step 1. Calculations for a normal year
        Step 2a. Space heating: calculate building thermal properties based on reference years
        Step 2b. Space heating: calculate heating time series
        Step 3. Clean and return dataframe
    
    
        Usage:
        farm_demand = run_scn_demand(smhi_folder, smhi_files, areas, energiprestande, water_share, cop_corr, ref_year, T_max, cc, cc_scen)
            where:
            - smhi_folder, smhi_files = path to files containing (hourly) temperature data
            - areas = list of bulding surface areas
            - energiprestande = list energy value from declarations for each building, given in kWh/m2/yr for a normal year
            - water_share = list of % of hot water production from energiprestande for each building
            - cop_corr = correction factors to convert old energiprestande calculations to actual heat consumed (efficiencies, COP, ...) 
            - ref_year = string, individual year, e.g. '2017' or list of string, e.g. ['2000', '2018'] representing a normal year
            - T_max = temperature above which heating is turned off on the farm, and demand is considered null, e.g. 17C in Swedish regulations
            - cc = True / False, if True, modifies temperature data to include climate change effect per seasons according to cc_scen
            - cc_scen = average temperature increase per season, from SMHI
        
    '''
    # Parse SMHI weather data
    s = clientSMHI()
    smhi = s.parseSMHI(smhi_folder, smhi_files)
    
    # Step 1. Calculations for a normal year
    tot_heat_demand, tot_heating_demand, tot_water_demand = BoverketToHeat(areas, energiprestande, water_share, cop_corr)
    
    # Step 2 & 3: Space heating
    # Space heating
    print("Reference year: ", ref_year)
    farm_demand = HeatingDemand(smhi, ref_year, tot_nrj = tot_heating_demand, T_max=T_max, cc=cc, cc_scen=cc_scen)
    
    # Step 3. Clean and return dataframe
    # Hot water
    w_kW = tot_water_demand / 365.25 / 24 # hourly vector: 365.25*24
    farm_demand['Water_kW']=np.repeat(w_kW, len(farm_demand))
      
    print("Returning farm_demand DataFrame")
      
    return farm_demand