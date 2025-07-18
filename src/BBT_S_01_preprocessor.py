# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 08:40:09 2020

@author: unterlass
"""

def concat():
    
    from BBT_S_00_utilities import utilities

    utils = utilities()
    
    ###########################################################################
    # import raw data and concat individual excel files to big parquet file/csv
    folder = r'01_raw_data\2 TBM Daten_EKS\S-1054_Rohdaten\all'
    
    drop_standstills = False  # change to true if standstills should be dropped
    check_for_miss_vals = False  # generates plot of missing values in dataset
    
    df = utils.concat_tables(folder, drop_standstills=drop_standstills,
                             check_for_miss_vals=check_for_miss_vals)
    
    print('\ndataset start - stop:', df['Station 1st Ref. point (projection on tunnel axis) m'].min(), '-', df['Station 1st Ref. point (projection on tunnel axis) m'].max())
    print('dataset end of Time:', df['Timestamp'].max())
    
    
    if drop_standstills is False:
        df.to_parquet(r'02_processed_data\00_TBMdata_BBT_S_wStandstills.gzip', index=False)
        #df.to_csv(fr'01_processed_data\00_TBMdata_wStandstills.csv', index=False)
    else:
        df.to_parquet(r'02_processed_data\01_TBMdata_BBT_S.gzip', index=False)
        #df.to_csv(fr'01_processed_data\00_TBMdata.csv', index=False)
    
    return df

df = concat()