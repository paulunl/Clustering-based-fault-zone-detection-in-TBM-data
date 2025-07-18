 # -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:25:05 2020

@author: unterlass
"""
def preprocessor():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from BBT_S_00_utilities import utilities, computation
    
    utils = utilities()
    comp = computation()
    
    ###########################################################################
    # load data
    df = pd.read_parquet(r'02_processed_data\01_TBMdata_BBT_S.gzip')
    df_ws = pd.read_parquet(r'02_processed_data\00_TBMdata_BBT_S_wStandstills.gzip')
    
    # drop all other columns
    df = df.loc[:,['Station 1st Ref. point (projection on tunnel axis) m [m]',
                   'Actual quantity of excavated material belt scale 1 [t]',
                   'Actual quantity of excavated material belt scale 2 [t]',
                   'Advance Force (mono/double shield) [kN]',
                   'Advance speed [mm/min]',
                   'Advance thrust force [kN]',
                   'Main drive torque [MNm]',
                   'Penetration [mm/rot]',
                   'Timestamp [yyyy-MM-dd HH:mm:ss]',
                   'energia specifica di scavo [MJ/m³]',
                   'thrust force single shield mode [kN]',
                   'velocità rotazione testa [rpm]',
                   'Advance number VMT (separate to ring number) [Unnamed: 73_level_1]',
                   'Advance number VMT (separate to ring number) [Unnamed: 78_level_1]',
                   'Total amount of power absorbed by TBM [kW]']]
    
    df_ws = df_ws.loc[:,['Station 1st Ref. point (projection on tunnel axis) m [m]',
                         'Actual quantity of excavated material belt scale 1 [t]',
                         'Actual quantity of excavated material belt scale 2 [t]',
                         'Advance Force (mono/double shield) [kN]',
                         'Advance speed [mm/min]',
                         'Advance thrust force [kN]',
                         'Main drive torque [MNm]',
                         'Penetration [mm/rot]',
                         'Timestamp [yyyy-MM-dd HH:mm:ss]',
                         'energia specifica di scavo [MJ/m³]',
                         'thrust force single shield mode [kN]',
                         'velocità rotazione testa [rpm]',
                         'Advance number VMT (separate to ring number) [Unnamed: 73_level_1]',
                         'Advance number VMT (separate to ring number) [Unnamed: 78_level_1]',
                         'Total amount of power absorbed by TBM [kW]']]

    df.rename(columns={
    'Station 1st Ref. point (projection on tunnel axis) m [m]': 'Tunnel Distance [m]',
    'Actual quantity of excavated material belt scale 1 [t]': 'Belt Scale 1 [t]',
    'Actual quantity of excavated material belt scale 2 [t]': 'Belt Scale 2 [t]',
    'Timestamp [yyyy-MM-dd HH:mm:ss]': 'Timestamp',
    'energia specifica di scavo [MJ/m³]': 'Specifc Energy [MJ/m³]',
    'thrust force single shield mode [kN]': 'Thrust Force Single Shield Mode [kN]',
    'velocità rotazione testa [rpm]': 'Cutterhead Rotations [rpm]',
    'Advance number VMT (separate to ring number) [Unnamed: 73_level_1]': 'Advance Number 1',
    'Advance number VMT (separate to ring number) [Unnamed: 78_level_1]': 'Advance Number 2',
    'Total amount of power absorbed by TBM [kW]': 'Power consumption [kW]'
    }, inplace=True)
    
    df_ws.rename(columns={
    'Station 1st Ref. point (projection on tunnel axis) m [m]': 'Tunnel Distance [m]',
    'Actual quantity of excavated material belt scale 1 [t]': 'Belt Scale 1 [t]',
    'Actual quantity of excavated material belt scale 2 [t]': 'Belt Scale 2 [t]',
    'Timestamp [yyyy-MM-dd HH:mm:ss]': 'Timestamp',
    'energia specifica di scavo [MJ/m³]': 'Specifc Energy [MJ/m³]',
    'thrust force single shield mode [kN]': 'Thrust Force Single Shield Mode [kN]',
    'velocità rotazione testa [rpm]': 'Cutterhead Rotations [rpm]',
    'Advance number VMT (separate to ring number) [Unnamed: 73_level_1]': 'Advance Number 1',
    'Advance number VMT (separate to ring number) [Unnamed: 78_level_1]': 'Advance Number 2',
    'Total amount of power absorbed by TBM [kW]': 'Power consumption [kW]'
    }, inplace=True)
    
    print('# datapoints without standstills', df['Tunnel Distance [m]'].count())
    
    # check NaNs
    nan_counts = df.isna().sum()
    nan_counts_ws = df_ws.isna().sum()

    # set NaNs to 99
    df[['Advance Force (mono/double shield) [kN]',
        'Advance thrust force [kN]',
        'Thrust Force Single Shield Mode [kN]',
        'Advance Number 1',
        'Advance Number 2']] = df[['Advance Force (mono/double shield) [kN]',
                                   'Advance thrust force [kN]',
                                   'Thrust Force Single Shield Mode [kN]',
                                   'Advance Number 1',
                                   'Advance Number 2']].fillna(0)
    
    df_ws[['Tunnel Distance [m]',
           'Belt Scale 1 [t]',
           'Belt Scale 2 [t]',
           'Advance speed [mm/min]',
           'Main drive torque [MNm]',
           'Penetration [mm/rot]',
           'Specifc Energy [MJ/m³]',
           'Cutterhead Rotations [rpm]',
           'Power consumption [kW]',           
        'Advance Force (mono/double shield) [kN]',
        'Advance thrust force [kN]',
        'Thrust Force Single Shield Mode [kN]',
        'Advance Number 1',
        'Advance Number 2']] = df_ws[['Tunnel Distance [m]',
                                      'Belt Scale 1 [t]',
                                   'Belt Scale 2 [t]',
                                   'Advance speed [mm/min]',
                                   'Main drive torque [MNm]',
                                   'Penetration [mm/rot]',
                                   'Specifc Energy [MJ/m³]',
                                   'Cutterhead Rotations [rpm]',
                                   'Power consumption [kW]', 
                                   'Advance Force (mono/double shield) [kN]',
                                   'Advance thrust force [kN]',
                                   'Thrust Force Single Shield Mode [kN]',
                                   'Advance Number 1',
                                   'Advance Number 2']].fillna(0)
    
    # delete remaining NaNs and reindex
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df_ws.dropna(inplace=True)
    df_ws.reset_index(drop=True, inplace=True)                  
    ###########################################################################
    # median for points with identical position
    df_median = df.groupby('Tunnel Distance [m]', as_index = False).median()
    print('\n# datapoints after grouping by Tunnel Distance', df_median['Tunnel Distance [m]'].count())
    
    '''
    # plot histogram for points with and without identical position
    df_grouped = df.groupby(['Tunnel Distance [m]'], as_index=True).agg(['count'])
    df_median_grouped = df_median.groupby(['Tunnel Distance [m]'], as_index=True).agg(['count']) #df_median.groupby(['Tunnel Distance [m]']).agg(['count']).index.values
    
    fig, (ax) = plt.subplots()
    ax.hist(df_grouped.index.values, 50, fc='darkgray', histtype='barstacked', ec='black')
    ax.hist(df_median_grouped.index.values, 50, fc='orange', histtype='barstacked', ec='black')
    ax.set_xlim(0, 6895)
    ax.set_xlabel('Tunnel Distance [m]')
    ax.set_ylabel('n', rotation=0)
    plt.tight_layout()
    plt.savefig(r'02_plots\hist_groupedbyTM.png', dpi=600)
    '''
    ###########################################################################
    # outlier filtering
    # features to be filtered
    mahal_features = ['Speed [mm/min]',
                      'Pressure A [bar]',
                      'Pressure B [bar]',
                      'Pressure C [bar]',
                      'Pressure D [bar]',
                      'CH Penetration [mm/rot]',
                      'CH Rotation [rpm]',
                      'CH Torque [MNm]',
                      'Thrust Force [kN]']
    
    # filter outliers with mahal distribution, for every datapoint with respect to the previous 100,
    # deleting the ones that lye out of the P85 percentile
    df_mahal = utils.filter_outliers(df_median, mahal_features, 85, 100)
    
    df_mahal = df_mahal.sort_values(by=['Tunnel Distance [m]'])
    df_mahal.index = np.arange(len(df_mahal))
    print('# datapoints after mahal distr.', df_mahal['Tunnel Distance [m]'].count())
    
    fig, (ax) = plt.subplots()
    ax.hist(df, 50, fc='darkgray', histtype='barstacked', ec='black')
    ax.hist(df_mahal, 50, fc='orange', histtype='barstacked', ec='black')
    ax.set_xlim(0, len(df))
    
    ###########################################################################
    # linear interpolation
    # difference in Tunnel Distance between single data points
    # interval = df_mahal['Tunnel Distance [m]'].diff().median()
    interval = df_median['Tunnel Distance [m]'].diff().median()
    interval = round(interval, 3)
    df_equal = utils.equally_spaced_df(df_median, 'Tunnel Distance [m]', interval)
    print('# datapoints after linear interp.', df_equal['Tunnel Distance [m]'].count())
    
    df = df_equal
    ###########################################################################
    # calculate spec. penetration
    penetration = df['Penetration [mm/rot]']
    
    # function to deal with multiple advance force columns
    # 'Advance Force (mono/double shield) [kN]',
    # 'Advance thrust force [kN]',
    # 'Thrust Force Single Shield Mode [kN]'
    # checks if two or more columns are non zero, in case two columns non zero
    # takes value in remaining column. in all other cases takes artihmetic mean
    
    def pick_or_average(row):
        values = [row['Advance Force (mono/double shield) [kN]'],
                  row['Advance thrust force [kN]'],
                  row['Thrust Force Single Shield Mode [kN]']]
        
        non_zero_values = [val for val in values if val != 0]
        
        if not non_zero_values:
            return 0
        elif len(non_zero_values) == 1:
            return non_zero_values[0]
        else:
            return sum(non_zero_values) / len(non_zero_values)
    
    tot_adv_force = df.apply(pick_or_average, axis=1)
    df['Advance Force [kN]'] = df.apply(pick_or_average, axis=1)

    df['Spec. Penetration [mm/rot/MN]'] = comp.s_pen(penetration, tot_adv_force)
    
    # calculate torque ratio
    penetration = df['Penetration [mm/rot]']
    tot_cutters = 41
    r_cutter = 482.6 #19" == 48.26cm
    # calculate M0 (torque needed for turning the cutting wheel when no advance)
    # caclulate adv. froce based on three columns with adv. force readings
    df_ws['Advance Force [kN]'] = df_ws.apply(pick_or_average, axis=1)
    
    df_M0 = df_ws.loc[
              (df_ws['Cutterhead Rotations [rpm]'] !=0) &
              (df_ws['Advance Force [kN]'] <= 0) &
              (df_ws['Main drive torque [MNm]'] > 0) &
              (df_ws['Main drive torque [MNm]'] < 14) &
              (df_ws['Penetration [mm/rot]'] > 0)
              ]
    
    M0 = df_M0['Main drive torque [MNm]'].median()*1000 # .mean()*1000
    # setting M0 to 175 kNm because average for 10m Diameter machines is 250kNm
    M0 = 175
    real_torque = df['Main drive torque [MNm]']*1000
    cutter_positions = 0.3 * 6.8 # 0.3 * diameter TBM CH
    
    df['torque ratio'], df['theoretical torque [MNm]'] = comp.t_ratio(
        tot_cutters, r_cutter, M0, tot_adv_force, penetration,
        real_torque, cutter_positions)    

    ###########################################################################
    #save df
    df = df.reset_index(drop=True)
    df = df.round({'Tunnel Distance [m]':3})
    df = df.dropna()
    df.to_parquet(r'02_processed_data\02_TBMdata_BBT_S_preprocessed.gzip', index=False)

    ###########################################################################
    return df

df = preprocessor()
