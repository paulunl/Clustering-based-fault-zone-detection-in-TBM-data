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
    # df_ws = pd.read_parquet(r'02_processed_data\00_TBMdata_BBT_S_wStandstills.gzip')
    
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
    
    # df_ws = df_ws.loc[:,['Station 1st Ref. point (projection on tunnel axis) m [m]',
    #                      'Actual quantity of excavated material belt scale 1 [t]',
    #                      'Actual quantity of excavated material belt scale 2 [t]',
    #                      'Advance Force (mono/double shield) [kN]',
    #                      'Advance speed [mm/min]',
    #                      'Advance thrust force [kN]',
    #                      'Main drive torque [MNm]',
    #                      'Penetration [mm/rot]',
    #                      'Timestamp [yyyy-MM-dd HH:mm:ss]',
    #                      'energia specifica di scavo [MJ/m³]',
    #                      'thrust force single shield mode [kN]',
    #                      'velocità rotazione testa [rpm]',
    #                      'Advance number VMT (separate to ring number) [Unnamed: 73_level_1]',
    #                      'Advance number VMT (separate to ring number) [Unnamed: 78_level_1]',
    #                      'Total amount of power absorbed by TBM [kW]']]

    df.rename(columns={
    'Station 1st Ref. point (projection on tunnel axis) m [m]': 'Tunnel Distance [m]',
    'Actual quantity of excavated material belt scale 1 [t]': 'Belt Scale 1 [t]',
    'Actual quantity of excavated material belt scale 2 [t]': 'Belt Scale 2 [t]',
    'Timestamp [yyyy-MM-dd HH:mm:ss]': 'Timestamp',
    'energia specifica di scavo [MJ/m³]': 'Specific Energy [MJ/m³]',
    'thrust force single shield mode [kN]': 'Thrust Force Single Shield Mode [kN]',
    'velocità rotazione testa [rpm]': 'Cutterhead Rotations [rpm]',
    'Advance number VMT (separate to ring number) [Unnamed: 73_level_1]': 'Advance Number 1',
    'Advance number VMT (separate to ring number) [Unnamed: 78_level_1]': 'Advance Number 2',
    'Total amount of power absorbed by TBM [kW]': 'Power consumption [kW]'
    }, inplace=True)
    
    # df_ws.rename(columns={
    # 'Station 1st Ref. point (projection on tunnel axis) m [m]': 'Tunnel Distance [m]',
    # 'Actual quantity of excavated material belt scale 1 [t]': 'Belt Scale 1 [t]',
    # 'Actual quantity of excavated material belt scale 2 [t]': 'Belt Scale 2 [t]',
    # 'Timestamp [yyyy-MM-dd HH:mm:ss]': 'Timestamp',
    # 'energia specifica di scavo [MJ/m³]': 'Specific Energy [MJ/m³]',
    # 'thrust force single shield mode [kN]': 'Thrust Force Single Shield Mode [kN]',
    # 'velocità rotazione testa [rpm]': 'Cutterhead Rotations [rpm]',
    # 'Advance number VMT (separate to ring number) [Unnamed: 73_level_1]': 'Advance Number 1',
    # 'Advance number VMT (separate to ring number) [Unnamed: 78_level_1]': 'Advance Number 2',
    # 'Total amount of power absorbed by TBM [kW]': 'Power consumption [kW]'
    # }, inplace=True)
    
    print('# datapoints without standstills', df['Tunnel Distance [m]'].count())
    
    # print('# datapoints with standstills', df_ws['Tunnel Distance [m]'].count())
    
    # check NaNs
    nan_counts = df.isna().sum()
    # nan_counts_ws = df_ws.isna().sum()
    
    # Calculate non-NaNs by subtracting from total row count
    non_nan_counts = len(df) - nan_counts
    
    # Combine both into a new DataFrame
    nan_summary = pd.DataFrame({
        'NaN_count': nan_counts,
        'Non-NaN_count': non_nan_counts
    })


    # set NaNs to xx
    df[['Advance Force (mono/double shield) [kN]',
        'Advance thrust force [kN]',
        'Thrust Force Single Shield Mode [kN]',
        'Advance Number 1',
        'Advance Number 2']] = df[['Advance Force (mono/double shield) [kN]',
                                   'Advance thrust force [kN]',
                                   'Thrust Force Single Shield Mode [kN]',
                                   'Advance Number 1',
                                   'Advance Number 2']].fillna(0)
    
    # df_ws[['Tunnel Distance [m]',
    #        'Belt Scale 1 [t]',
    #        'Belt Scale 2 [t]',
    #        'Advance speed [mm/min]',
    #        'Main drive torque [MNm]',
    #        'Penetration [mm/rot]',
    #        'Specific Energy [MJ/m³]',
    #        'Cutterhead Rotations [rpm]',
    #        'Power consumption [kW]',           
    #     'Advance Force (mono/double shield) [kN]',
    #     'Advance thrust force [kN]',
    #     'Thrust Force Single Shield Mode [kN]',
    #     'Advance Number 1',
    #     'Advance Number 2']] = df_ws[['Tunnel Distance [m]',
    #                                   'Belt Scale 1 [t]',
    #                                'Belt Scale 2 [t]',
    #                                'Advance speed [mm/min]',
    #                                'Main drive torque [MNm]',
    #                                'Penetration [mm/rot]',
    #                                'Specific Energy [MJ/m³]',
    #                                'Cutterhead Rotations [rpm]',
    #                                'Power consumption [kW]', 
    #                                'Advance Force (mono/double shield) [kN]',
    #                                'Advance thrust force [kN]',
    #                                'Thrust Force Single Shield Mode [kN]',
    #                                'Advance Number 1',
    #                                'Advance Number 2']].fillna(0)
    
    # delete remaining NaNs and reindex
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # df_ws.dropna(inplace=True)
    # df_ws.reset_index(drop=True, inplace=True)                  
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
    ax.set_xlim(13600, 27000)
    ax.set_xlabel('Tunnel Distance [m]')
    ax.set_ylabel('n', rotation=0)
    plt.tight_layout()
    plt.savefig(r'03_plots\hist_groupedbyTM.png', dpi=600)
    '''
    ###########################################################################
    # outlier filtering
    # features to be filtered
    # mahal_features = ['Speed [mm/min]',
    #                   'Pressure A [bar]',
    #                   'Pressure B [bar]',
    #                   'Pressure C [bar]',
    #                   'Pressure D [bar]',
    #                   'CH Penetration [mm/rot]',
    #                   'CH Rotation [rpm]',
    #                   'CH Torque [MNm]',
    #                   'Thrust Force [kN]']
    
    # # filter outliers with mahal distribution, for every datapoint with respect to the previous 100,
    # # deleting the ones that lye out of the P85 percentile
    # df_mahal = utils.filter_outliers(df_median, mahal_features, 85, 100)
    
    # df_mahal = df_mahal.sort_values(by=['Tunnel Distance [m]'])
    # df_mahal.index = np.arange(len(df_mahal))
    # print('# datapoints after mahal distr.', df_mahal['Tunnel Distance [m]'].count())
    
    # fig, (ax) = plt.subplots()
    # ax.hist(df, 50, fc='darkgray', histtype='barstacked', ec='black')
    # ax.hist(df_mahal, 50, fc='orange', histtype='barstacked', ec='black')
    # ax.set_xlim(0, len(df))
    
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
    # hard drop of outliers beyond machine limits
    df.drop(df[df['Main drive torque [MNm]'] > 14].index, inplace=True)
    
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
    
    # drop other advance force columns
    df.drop(['Advance Force (mono/double shield) [kN]',
             'Advance thrust force [kN]',
             'Thrust Force Single Shield Mode [kN]'], axis=1, inplace=True)
    
    # hard drop of outlier beyond machine limit
    df.drop(df[df['Advance Force [kN]'] > 42750].index, inplace=True)

    df['Spec. Penetration [mm/rot/MN]'] = comp.s_pen(penetration, tot_adv_force)
    
    # calculate torque ratio
    penetration = df['Penetration [mm/rot]']
    tot_adv_force = df['Advance Force [kN]']
    tot_cutters = 41 # Number of Cutters
    r_cutter = 482.6 #19" == 48.26cm Radius Disc Cutter
    
    '''
    # calculate M0 (torque needed for turning the cutting wheel when no advance)
    # caclulate adv. force based on three columns with adv. force readings
    df_ws['Advance Force [kN]'] = df_ws.apply(pick_or_average, axis=1)
    df_M0 = df_ws.loc[
              (df_ws['Cutterhead Rotations [rpm]'] !=0) &
              (df_ws['Advance Force [kN]'] <= 0) &
              (df_ws['Main drive torque [MNm]'] > 0) &
              (df_ws['Main drive torque [MNm]'] < 14) &
              (df_ws['Penetration [mm/rot]'] > 0)
              ]
    M0 = df_M0['Main drive torque [MNm]'].median()*1000 # .mean()*1000
    '''
    # setting M0 to 175 kNm because average for 10m Diameter machines is 250kNm
    M0 = 175
    real_torque = df['Main drive torque [MNm]']*1000
    cutter_positions = 0.3 * 6.8 # 0.3 * diameter TBM CH
    
    df['torque ratio'], df['theoretical torque [MNm]'] = comp.t_ratio(
        tot_cutters, r_cutter, M0, tot_adv_force, penetration,
        real_torque, cutter_positions)
    
    ###########################################################################
    # add geoogical information
    file = (r'01_raw_data\3 Geo Daten_EKS\Tunnelbänder_Vorabzug\20250722_Geo-Daten.xlsx')
            
    # Define clean column names based on the provided image
    columns = [
        "Homogen Area",
        "Start [km]",
        "End [km]",
        "Length [m]",
        "Gebirgsart",
        "OB Aufnahme #",
        "OB Aufnahme km",
        "RMR",
        "UCS [MPa] min",
        "UCS [MPa] max",
        "RQD [%]",
        "GSI",
        "TCR GVT min",
        "TCR GVT max",
        "Dip/Dir [°/°]",
        "Spacing [mm]",
        "Class"
    ]
    
    # Define the exact columns to keep
    columns_to_keep = [
        "OB Aufnahme km",      
        "RMR",
        "UCS [MPa] min",     
        "UCS [MPa] max", 
        "TCR GVT min",
        "TCR GVT max",         
        "Class"
    ]

    # Read the data, skip header rows, assign column names
    df_labels = pd.read_excel(file, sheet_name='CE-EKS_Geo-Daten', skiprows=3,
                              skipfooter=3, header=None)
    df_labels.columns = columns
    
    # Keep only the selected columns
    df_labels = df_labels[columns_to_keep]

    # Drop fully empty rows
    df_labels.dropna(how='all', inplace=True)
    
    # set common index for merge
    df['Tunnel Distance [m]'] = np.round(df.index.astype(float), 3)
    df.set_index('Tunnel Distance [m]', inplace=True, drop=False)
    # set index to float
    df.index = df.index.astype(float)
    df_labels.set_index("OB Aufnahme km", inplace=True)
    df_labels.index = df_labels.index.astype(float)
    
    # Define mapping from Roman numerals to integers
    class_map = {
        'I': 1,
        'II': 2,
        'III': 3,
        'IV': 4,
        'V': 5
    }
    
    # Apply the mapping to the 'Class' column
    df_labels['Class'] = df_labels['Class'].map(class_map)
    
    # To replace all instances of a dash ('-') with NaN (missing values)
    df_labels.replace('-', np.nan, inplace=True)

    # function to add geological information +-10m around each geological inspection
    def fill_nearby_values(df1, df2, column_name, window=10):
        """
        Fills a new column in df1 using values from df2[column_name], 
        spreading each known value to ±window meters based on the index ('tunnel distance [m]').
    
        Parameters:
        - df1: DataFrame where the new column will be added
        - df2: DataFrame that contains sparse values in column_name
        - column_name: str, name of the column in df2 to be copied to df1
        - window: int, the ± meter range to fill around each known value
    
        Returns:
        - A pandas Series with the new column values aligned to df1's index
        """
        df1 = df1.copy()
        df1 = df1.sort_index()
        df2 = df2.sort_index()
    
        # Initialize new column with NaN
        filled_column = pd.Series(index=df1.index, dtype='float64')
    
        # Fill values from df2 into df1 within the ±window range
        for dist, value in df2[column_name].dropna().items():
            mask = (df1.index >= dist - window) & (df1.index <= dist + window)
            filled_column[mask] = value
    
        return filled_column
    
    df['RMR'] = fill_nearby_values(df, df_labels, 'RMR', window=10)
    df['UCS [MPa] min'] = fill_nearby_values(df, df_labels, 'UCS [MPa] min', window=10)
    df['UCS [MPa] max'] = fill_nearby_values(df, df_labels, 'UCS [MPa] max', window=10)
    df['TCR GVT min'] = fill_nearby_values(df, df_labels, 'TCR GVT min', window=10)
    df['TCR GVT max'] = fill_nearby_values(df, df_labels, 'TCR GVT max', window=10)
    df['Class'] = fill_nearby_values(df, df_labels, 'Class', window=10)
    
    # fill NaNs with 0
    df.fillna(0, inplace=True)

    ###########################################################################
    # add fault zones
    file_faults = (r'01_raw_data\3 Geo Daten_EKS\Tunnelbänder_Vorabzug\Faultzones.xlsx')

    # Read the data, skip header rows, assign column names
    df_faults = pd.read_excel(file_faults, sheet_name='Tabelle1')
    
    # set common index for merge
    df_faults.set_index("To", inplace=True)
    
    # merge dfs
    df = pd.merge(df, df_faults,
                  left_index=True, right_index=True, how='outer')
    
    df['Fault'].fillna(method='bfill', inplace=True)

    #save df
    # df = df.reset_index(drop=True)
    df = df.dropna()
    df.to_parquet(r'02_processed_data\02_TBMdata_BBT_S_preprocessed_wlabels.gzip', index=False)

    ###########################################################################
    return df

df = preprocessor()
