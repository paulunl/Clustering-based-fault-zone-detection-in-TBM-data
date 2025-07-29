# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 08:43:26 2020

@author: erharter
"""


# script with utilities, formulas etc

from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import pandas as pd
import PIL
from scipy import interpolate
from tqdm import tqdm

class utilities:

    def __init__(self):
        self.previous_advance_force_col = None
        pass
    
    
    def equally_spaced_df(self, df, tunnellength, interval):
        min_length = df[tunnellength].min()
        max_length = df[tunnellength].max()

        equal_range = np.arange(min_length, max_length, interval)

        df_interp = pd.DataFrame({tunnellength: equal_range})

        for feature in df.drop(tunnellength, axis=1).columns:
            f = interpolate.interp1d(df[tunnellength], df[feature],
                                     kind='linear')
            df_interp[feature] = f(equal_range)

        df_interp.set_index(tunnellength, drop=False, inplace=True)

        return df_interp
    
    '''
    def equally_spaced_df(self, df, tunnellength, interval):
        min_length = tunnellength.min()
        max_length = tunnellength.max()

        equal_range = np.arange(min_length, max_length, interval)

        df_interp = pd.DataFrame({tunnellength: equal_range})

        for feature in df.drop(tunnellength, axis=1).columns:
            f = interpolate.interp1d(df[tunnellength], df[feature],
                                     kind='linear')
            df_interp[feature] = f(equal_range)

        df_interp.set_index(tunnellength, drop=False, inplace=True)

        return df_interp
    '''
    def check_for_miss_vals(self, df):
        print(df['Timestamp'])
        missing = df['Timestamp'].diff()
        missing = missing / np.timedelta64(1, 's')
        missing_idxs = np.where(missing > (60 * 60))[0]

        # print intervals where more than 1h of data are missing
        for missing_idx in missing_idxs:
            start = df['Timestamp'].iloc[missing_idx-1]
            stop = df['Timestamp'].iloc[missing_idx]
            print(f'missing data between {start} & {stop}')

        fig, ax = plt.subplots()
        ax.plot(df['Timestamp'], (missing / 60 / 60))
        ax.grid()
        ax.set_ylabel('missing data [h]')
        ax.tick_params(axis='x', labelrotation=45, )
        plt.tight_layout()

    def concat_tables(self, folder, drop_standstills=True,
                      check_for_miss_vals=True):
        filenames = []
        for file in listdir(folder):
            if file.split('.')[1] == 'csv':
                filenames.append(file)

        files = []

        for i, file in enumerate(filenames[:500000]):

            df = pd.read_csv(fr'{folder}\{file}',
                             sep=';',
                             skiprows = 0,
                             header = [1, 2],
                             parse_dates=[0]
                             )
            
            # strip trailing (and leading) whitespace from multi-index column headers
            df.columns = pd.MultiIndex.from_tuples(
                [(str(col[0]).strip(), str(col[1]).strip()) for col in df.columns]
                )

            print(file)

            # hard drop of most obvious outliers
            # df.drop(df[df['Tunnel Distance [m]'] > 7000].index, inplace=True)
            
            # convert strings to numeric in order to get NaNs instead of strings (e.g., \N)
            df = df.apply(pd.to_numeric, errors='coerce')
            # df['Adv Activated [-]'] = pd.to_numeric(df['Adv Activated [-]'], errors='coerce')
            # df['Tunnel Distance [m]'] = pd.to_numeric(df['Tunnel Distance [m]'], errors='coerce')
            # df['CH Torque [MNm]'] = pd.to_numeric(df['CH Torque [MNm]'], errors='coerce')
            # df['Thrust Force [kN]'] = pd.to_numeric(df['Thrust Force [kN]'], errors='coerce')
            # df['CH Rotation [rpm]'] = pd.to_numeric(df['CH Rotation [rpm]'], errors='coerce')
            # df['CH Penetration [mm/rot]'] = pd.to_numeric(df['CH Penetration [mm/rot]'], errors='coerce')
            
            # replace NaN with 0 in order to be able to drop standstills, because '<=' not supported between instances of 'str' and 'int'
            # df['Tunnel Distance [m]'] = df['Tunnel Distance [m]'].fillna(0)
            # df['CH Torque [MNm]'] = df['CH Torque [MNm]'].fillna(0)
            # df['Thrust Force [kN]'] = df['Thrust Force [kN]'].fillna(0)
            # df['CH Rotation [rpm]'] = df['CH Rotation [rpm]'].fillna(0)
            # df['CH Penetration [mm/rot]'] = df['CH Penetration [mm/rot]'].fillna(0)
                    
            if drop_standstills is True:
                # Try all possible labels for advance force
                advance_force_options = [
                    ('Advance Force (mono/double shield)', 'kN'),
                    ('Advance thrust force', 'kN'),
                    ('thrust force single shield mode', 'kN'),]
                
                # Find all columns from options present in df
                matched_cols = [col for col in advance_force_options if col in df.columns]
                
                if not matched_cols:
                    raise KeyError("Advance force columns not found in DataFrame.")
                
                if len(matched_cols) > 1:
                    print(f"Warning: Multiple advance force columns found: {matched_cols}.")
                    # You can decide to pick the first:
                    advance_force_col = matched_cols[0]
                    print(f"Using {advance_force_col} by default.")
                else:
                    advance_force_col = matched_cols[0]
                
                if advance_force_col != self.previous_advance_force_col:
                    print(f"Advance force column changed from {self.previous_advance_force_col} to: {advance_force_col}")
                    self.previous_advance_force_col = advance_force_col
                    
                # advance_force_col = None
                # for col in advance_force_options:
                #     if col in df.columns:
                #         advance_force_col = col
                #         break
            
                # if advance_force_col is None:
                #     raise KeyError("Advance force column not found in DataFrame.")
            
                idx_standstill = df.loc[
                    (df[('Main drive torque', 'MNm')] <= 0)
                    | (df[advance_force_col] <= 0)
                    | (df[('velocità rotazione testa', 'rpm')] <= 0)
                    | (df[('Penetration', 'mm/rot')] <= 0)
                ].index
            
                df.drop(idx_standstill, inplace=True)

            files.append(df)
            print(f'{i} / {len(filenames)-1} csv done')  # status

        df = pd.concat(files, sort=True)
        # flatten column names
        df.columns = ['{} [{}]'.format(name, unit) for name, unit in df.columns]

        # try:
        #     df.drop(['Tunnel Distance [m]'], axis=1, inplace=False)
        # except KeyError:
        #     pass
        # df.dropna(inplace=True)
        
        # set dateformate to european add exception handling for varying datetime formats
        # try:
        #     # Try parsing as DD.MM.YYYY HH:MM:SS
        #     df['Timestamp'] = pd.to_datetime(df['Timestamp'].str.strip(), format='%d.%m.%Y %H:%M:%S')
        # except (ValueError, AttributeError):
        #     # If that fails, parse as Unix nanoseconds
        #     df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ns')
        
        # df.sort_values(by='Timestamp', inplace=True)

        # check for missing values in time series
        if drop_standstills is False and check_for_miss_vals is True:
            self.check_for_miss_vals(df)

        return df

    def filter_outliers(self, df, mahal_features, percentile, length):
        drop_idxs = []

        for i in tqdm(df.index):
            #if i % 10000 == 0:
                #print(f'{i} / {len(df)}')
            if i > length and i <= len(df):
                window = df[i-length: i]
                # calcs the mahalanobis dist. for every point based on features
                comp = computation()
                try:
                    mahal = comp.MahalanobisDist(window[mahal_features].values)
                    thresh = np.percentile(mahal, percentile)
                    drop_idx = np.where(mahal > thresh)[0]
                    drop_idxs.append(window.index[drop_idx].values)
                except TypeError:
                    pass

        drop_idxs = np.concatenate(drop_idxs)
        df.drop(drop_idxs, inplace=True)
        df.index = np.arange(len(df))

        return df

    def all_combinations(self, features):
        max_features = len(features) + 1

        all_combs = []

        for num in range(1, max_features):
            for i in combinations(features, num):
                all_combs.append(list(i))

        return all_combs

    def add_GAs(self, df, filepath):
        try:
            # add Gebirgsarten
            df_GA = pd.read_excel(filepath, sheet_name=0)

            GAs = ['6d-1', '6d-2', '6e', '6f', '8c', '8d']
            # surpress SettingWithCopyWarning
            pd.options.mode.chained_assignment = None  # default='warn'
            for GA in GAs:
                #print(GA)
                idx = np.where(df_GA[GA].notnull() == True)[0]
                df_GA[GA].iloc[idx] = GA

            # merge on not nan values
            df_GA['GAs'] = df_GA[GAs].bfill(axis=1).iloc[:, 0]

            df_GA.set_index('bis', inplace=True)

            # drop all columns but GA columns
            df_GA.drop(df_GA.drop(['GAs'], axis=1).columns,
                       axis=1, inplace=True)

            df_GA.replace({'GAs': {'6d-1': 1, '6d-2': 2, '6e': 3,
                                   '6f': 4, '8c': 5, '8d': 6}}, inplace=True)

            df = pd.merge(df, df_GA,
                          left_index=True, right_index=True, how='outer')

            df['GAs'].fillna(method='bfill', inplace=True)
            return df

        except FileNotFoundError:
            pass

    def add_SVs(self, df, filepath):
        try:
            # add Systemverhalten
            df_SV = pd.read_excel(filepath, sheet_name='SV-I')

            df_SV.set_index('bis', inplace=True, drop=False)
            df_SV.drop(['von', 'bis', 'lg'], axis=1, inplace=True)
            try:
                df_SV.replace({'SV': {'Kein SV': np.nan}}, inplace=True)
            except TypeError:
                pass

            df = pd.merge(df, df_SV,
                          left_index=True, right_index=True, how='outer')

            df['SV'].fillna(method='bfill', inplace=True)
            return df

        except FileNotFoundError:
            pass

    def to_categorical(self, y_train, num_classes):
        b = np.zeros((len(y_train), num_classes))
        b[np.arange(len(y_train)), y_train] = 1
        return b

###############################################################################


class plotter:

    def __init__(self):
        pass

    def add_line(self, df, x, y, roll_mean=True):
        ax = plt.gca()

        if roll_mean is True:
            ax.plot(df[x], df[y], alpha=0.6, color='grey')
            ax.plot(df[x], df[y].rolling(window=150, center=True).mean(),
                    color='black')
        else:
            ax.plot(df[x], df[y], color='black')

        ax.set_ylabel(y)
        ax.grid()
        ax.set_xlim(left=df[x].min(), right=df[x].max())

    def add_hist(self, df, x):
        ax = plt.gca()
        ax.hist(df[x], color='grey', edgecolor='black')

    def add_classification(self, df, x, kind, ylabel, xlabel, classes, FROM, TO,
                           Q_class_letters):
        ax = plt.gca()

        ax.plot(df[x], df[kind], color='grey', zorder=1)
        ax.set_xlim(left=FROM, right=TO)
        colors = plt.cm.get_cmap('Paired')
        colors = colors(np.linspace(0, 1, num=len(classes)))

        for i, c in enumerate(np.arange(1, len(classes)+1)):
            idx = np.where(df[kind] == c)[0]
            ax.scatter(df[x].iloc[idx], df[kind].iloc[idx], color=colors[i],
                       zorder=10)
        ax.set_xlim(left=FROM, right=TO)

        ax.set_ylabel(ylabel, rotation=90)
        ax.set_xlabel(xlabel)
        ax.yaxis.set_ticks(np.arange(1, len(classes)+1))
        ax.set_yticklabels(Q_class_letters)
        ax.set_ylim(bottom=0.5, top=len(classes)+0.5)
        ax.set_facecolor(color=(0.95, 0.95, 0.95))
        ax.grid(alpha=0.5)

    def add_scatter(self, x, y, df):
        ax = plt.gca()
        classes = np.arange(1, 7)

        colors = plt.cm.get_cmap('Paired')
        colors = colors(np.linspace(0, 1, num=len(classes)))

        for i, Q_class in enumerate(classes):
            idx = np.where(df['Q_class'] == Q_class)[0]
            ax.scatter(df[x].iloc[idx], df[y].iloc[idx], color=colors[i],
                       edgecolor='black', alpha=0.1)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(alpha=0.5, zorder=0)

    def boxplot(self, data, y, features, classification, savepath):

        try:
            if y.shape[1] > 1:
                y = np.argmax(y, axis=1)
        except IndexError:
            pass

        data = data / data.max(axis=0)

        values = []

        for typ in np.unique(y):
            idx = np.where(y == typ)[0]
            vals = np.take(data, idx, axis=0)
            print(typ, vals.shape)
            values.append(vals)

        fig = plt.figure(figsize=(len(np.unique(y))*2.5, 5))

        for i in range(len(classification)):
            ax = fig.add_subplot(1, len(classification)/1, i+1)
            try:
                ax.boxplot(values[i], whis=[5, 95], showfliers=False, zorder=10)
                medians = np.median(values[i], axis=0)
                ax.plot(range(1, len(medians)+1), medians,
                        color='grey', alpha=0.5, zorder=1)
                perc_tm = round(len(values[i]) / len(data) * 100, 2)
                ax.set_title(f'{classification[i]}\nStrecke: {perc_tm}%')
            except IndexError:
                pass

            ax.set_xticklabels(features, rotation=90, 
                               horizontalalignment='center')

            ax.set_ylim(top=1, bottom=0)
            ax.grid(alpha=0.5)
        plt.tight_layout()
        plt.savefig(savepath, dpi=600)

    def create_coords(self, x_labels, y_labels, x_name, y_name, df):
        x, y = np.arange(0, len(x_labels)+1), np.arange(1, len(y_labels)+1)
        xv, yv = np.meshgrid(x, y)
        xv, yv = xv.flatten(), yv.flatten()

        zs = []

        for i in range(len(xv)):
            idxs = np.where((df[x_name] == xv[i]) & (df[y_name] == yv[i]))[0]
            zs.append(len(idxs))

        idxs = np.where(np.array(zs) == 0)[0]
        xs = np.delete(xv, idxs)
        ys = np.delete(yv, idxs)
        zs = np.delete(zs, idxs)

        zs = zs.astype('int32')
        print(zs.sum())
        return xs, ys, zs
    
    def AGT21_logo(self, width, height):
        im = PIL.Image.open(r'00_raw_data\Logo_AGT21.png')
        im = im.resize((width, height))  # ratio 150 / 100
        im_width, im_height = im.size
        im = np.array(im).astype(np.float) / 255
        im = np.flip(im, axis=0)
        return im

###############################################################################


class computation:

    def __init__(self):
        pass

    def MahalanobisDist(self, data):
        data = np.transpose(data)
        #print(data.shape)
        n_dims = data.shape[0]
        # calculate the covariance matrix
        covariance_xyz = np.cov(data)
        # take the inverse of the covariance matrix
        try:
            inv_covariance_xyz = np.linalg.inv(covariance_xyz)

            means = []
            for i in range(n_dims):
                means.append(np.mean(data[i]))

            diffs = []
            for i in range(n_dims):
                diff = np.asarray([x_i - means[i] for x_i in data[i]])
                diffs.append(diff)
            diffs = np.transpose(np.asarray(diffs))

            # calculate the Mahalanobis Distance for each data sample
            md = []
            for i in range(len(diffs)):
                md.append(np.sqrt(np.dot(np.dot(np.transpose(diffs[i]),
                                                inv_covariance_xyz),
                                  diffs[i])))
            return md

        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print('singular matrix')
                pass
            else:
                raise

    def s_en(self, thrust, area, rotations, torque, penetration):
        # computes the specific energy after Teale 1965
        f = thrust / area
        x = (2 * np.pi * rotations * torque) / (area * penetration)
        return (f + x)

    def s_pen(self, penetration, adv_force):
        # computes specific penetration
        spec_penetration = penetration / (adv_force / 1000)
        return spec_penetration

    def t_ratio(self, tot_cutters, r_cutter, M0,
                tot_adv_force, penetration,
                real_torque, cutter_positions):
        # computes the torque ratio after Radoncic 2014
        length = len(tot_adv_force)

        # cutter radius
        r_cutter = np.full(tot_cutters, r_cutter)
        r_cutter = np.meshgrid(r_cutter, np.arange(length))[0]

        # avg. normal force per cutter
        Fn = (tot_adv_force)/tot_cutters
        Fn = np.meshgrid(np.arange(tot_cutters), Fn)[1]

        # cutting angle
        penetration = np.meshgrid(np.arange(tot_cutters),
                                  (penetration))[1]
        angle = np.degrees(np.arccos((r_cutter - penetration)/r_cutter))
        # avg. tangential force:
        Ft = Fn * np.tan(np.deg2rad(angle)/2)
        cutter_positions = np.meshgrid(cutter_positions, np.arange(length))[0]
        sums = (np.sum((cutter_positions * Ft), axis=1) + M0)           
        print(sums)
        theoretical_torque = sums

        torque_ratio = real_torque / theoretical_torque

        return torque_ratio, theoretical_torque

    def s_friction(self, thrust, cutterhead_force, tension_backupsys):
        return thrust - cutterhead_force - tension_backupsys
