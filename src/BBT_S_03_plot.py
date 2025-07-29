# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:47:56 2020

@author: unterlass
"""
import pandas as pd
    
df = pd.read_parquet(r'02_processed_data\02_TBMdata_BBT_S_preprocessed.gzip')
 
def plot(df, FROM, TO, WINDOW):
        import matplotlib.pyplot as plt
        import numpy as np
        import statistics as s
        
        ######################################################################
        # plot thrust force
        fig, (ax4, ax5, ax1, ax2, ax3, ax6, ax7) = plt.subplots(nrows=7, ncols=1, figsize=(16, 10))
        
        ax1.plot(df['Tunnel Distance [m]'], df['Advance Force [kN]'], color='grey',
                 alpha=0.6)
        ax1.plot(df['Tunnel Distance [m]'], df['Advance Force [kN]'].rolling(window=WINDOW, center=True).mean(),
                 color='black', linewidth=0.5)
        
        ax1.axhline(y=df['Advance Force [kN]'].mean() + 2*s.stdev(df['Advance Force [kN]']),
                    color='black', alpha=0.9, linestyle='--', linewidth=0.5, label='+/- 2σ')
        ax1.axhline(y=df['Advance Force [kN]'].mean() - 2*s.stdev(df['Advance Force [kN]']),
                    color='black', alpha=0.9, linestyle='--', linewidth=0.5)
        ax1.set_xticklabels([])
        ax1.set_xlim(FROM, TO)
        ax1.set_ylim(0, 18000)
        ax1.set_ylabel('Advance Force\n'
                       ' [kN]', rotation=90)
        ax1.grid(alpha=0.5)
        ax1.legend(loc=4, prop={'size': 10})
        
        # plot torque ratio
        ax2.plot(df['Tunnel Distance [m]'], df['torque ratio'], color='grey',
                 alpha=0.6)
        ax2.plot(df['Tunnel Distance [m]'], df['torque ratio'].rolling(window=WINDOW, center=True).mean(),
                 color='black', linewidth=0.5)
        
        ax2.axhline(y=df['torque ratio'].mean() + 2*s.stdev(df['torque ratio']),
                    color='black', alpha=0.9, linestyle='--', linewidth=0.5, label='+/- 2σ')
        ax2.axhline(y=df['torque ratio'].mean() - 2*s.stdev(df['torque ratio']),
                    color='black', alpha=0.9, linestyle='--', linewidth=0.5)
        ax2.set_xticklabels([])
        ax2.set_xlim(FROM, TO)
        ax2.set_ylim(0, 1.2)
        #ax2.set_ylim(ax2.get_ylim()[::-1])
        
        ax2.set_ylabel('torque ratio', rotation=90)
        ax2.set_yticks([0, 1])
        ax2.grid(alpha=0.5)
        ax2.legend(loc=4, prop={'size': 10})

        # plot spec. penetration
        ax3.plot(df['Tunnel Distance [m]'], df['Spec. Penetration [mm/rot/MN]'],
                    color='grey', alpha=0.6)
        ax3.plot(df['Tunnel Distance [m]'],
                 df['Spec. Penetration [mm/rot/MN]'].rolling(window=WINDOW, center=True).mean(),
                 color='black', linewidth=0.5)
        ax3.axhline(y=df['Spec. Penetration [mm/rot/MN]'].mean() + 2*s.stdev(df['Spec. Penetration [mm/rot/MN]']),
                    color='black', alpha=0.9, linestyle='--', linewidth=0.5, label='+/- 2σ')
        ax3.axhline(y=df['Spec. Penetration [mm/rot/MN]'].mean() - 2*s.stdev(df['Spec. Penetration [mm/rot/MN]']),
                    color='black', alpha=0.9, linestyle='--', linewidth=0.5)  
        ax3.set_xlim(FROM, TO)
        ax3.set_ylim(0, 5)
        ax3.set_ylabel('Spec. Pene.\n'
                       ' [mm/rot/MN]', rotation=90)
        ax3.set_yticks([0, 2.5, 5])
        ax3.set_xticklabels([])
        ax3.legend(loc=4, prop={'size': 10})
        ax3.grid(alpha=0.5)
        
        # plot penetration
        ax4.plot(df['Tunnel Distance [m]'], df['Penetration [mm/rot]'],
                    color='grey', alpha=0.6)
        ax4.plot(df['Tunnel Distance [m]'],
                 df['Penetration [mm/rot]'].rolling(window=WINDOW, center=True).mean(),
                 color='black', linewidth=0.5)
        ax4.axhline(y=df['Penetration [mm/rot]'].mean() + 2*s.stdev(df['Penetration [mm/rot]']),
                    color='black', alpha=0.9, linestyle='--', linewidth=0.5, label='+/- 2σ')
        ax4.axhline(y=df['Penetration [mm/rot]'].mean() - 2*s.stdev(df['Penetration [mm/rot]']),
                    color='black', alpha=0.9, linestyle='--', linewidth=0.5)     
        ax4.set_xlim(FROM, TO)
        ax4.set_ylim(0, 60)
        ax4.set_ylabel('Penetration\n'
                       ' [mm/rot]', rotation=90)
        ax4.set_yticks([20, 40, 60])
        ax4.set_xticklabels([])
        ax4.legend(loc=4, prop={'size': 10})
        ax4.grid(alpha=0.5)
        
        # plot torque
        ax5.plot(df['Tunnel Distance [m]'], df['Main drive torque [MNm]'],
                    color='grey', alpha=0.6)
        ax5.plot(df['Tunnel Distance [m]'],
                 df['Main drive torque [MNm]'].rolling(window=WINDOW, center=True).mean(),
                 color='black', linewidth=0.5)
        ax5.axhline(y=df['Main drive torque [MNm]'].mean() + 2*s.stdev(df['Main drive torque [MNm]']),
                    color='black', alpha=0.9, linestyle='--', linewidth=0.5, label='+/- 2σ')
        ax5.axhline(y=df['Main drive torque [MNm]'].mean() - 2*s.stdev(df['Main drive torque [MNm]']),
                    color='black', alpha=0.9, linestyle='--', linewidth=0.5)     
        ax5.set_xlim(FROM, TO)
        ax5.set_ylim(0, 4)
        ax5.set_ylabel('Main drive torque\n'
                       ' [MNm]', rotation=90)
        ax5.set_yticks([0, 2, 4])
        ax5.set_xticklabels([])
        ax5.legend(loc=4, prop={'size': 10})
        ax5.grid(alpha=0.5)
        
        # plot specific energy
        ax6.plot(df['Tunnel Distance [m]'], df['Specifc Energy [MJ/m³]'],
                    color='grey', alpha=0.6)
        ax6.plot(df['Tunnel Distance [m]'],
                 df['Specifc Energy [MJ/m³]'].rolling(window=WINDOW, center=True).mean(),
                 color='black', linewidth=0.5)
        ax6.axhline(y=df['Specifc Energy [MJ/m³]'].mean() + 2*s.stdev(df['Specifc Energy [MJ/m³]']),
                    color='black', alpha=0.9, linestyle='--', linewidth=0.5, label='+/- 2σ')
        ax6.axhline(y=df['Specifc Energy [MJ/m³]'].mean() - 2*s.stdev(df['Specifc Energy [MJ/m³]']),
                    color='black', alpha=0.9, linestyle='--', linewidth=0.5)     
        ax6.set_xlim(FROM, TO)
        ax6.set_ylim(0, 100)
        ax6.set_ylabel('Specifc Energy\n'
                       ' [MJ/m³]', rotation=90)
        ax6.set_yticks([0, 50, 100])
        ax6.set_xticklabels([])
        ax6.legend(loc=4, prop={'size': 10})
        ax6.grid(alpha=0.5)

        # plot belt scale
        ax7.plot(df['Tunnel Distance [m]'], df['Belt Scale 1 [t]'],
                    color='grey', alpha=0.6)
        ax7.plot(df['Tunnel Distance [m]'],
                 df['Belt Scale 1 [t]'].rolling(window=WINDOW, center=True).mean(),
                 color='black', linewidth=0.5)
        ax7.axhline(y=df['Belt Scale 1 [t]'].mean() + 2*s.stdev(df['Belt Scale 1 [t]']),
                    color='black', alpha=0.9, linestyle='--', linewidth=0.5, label='+/- 2σ')
        ax7.axhline(y=df['Belt Scale 1 [t]'].mean() - 2*s.stdev(df['Specifc Energy [MJ/m³]']),
                    color='black', alpha=0.9, linestyle='--', linewidth=0.5)     
        ax7.set_xlim(FROM, TO)
        ax7.set_ylabel('Tunnel Distance [m]')
        ax7.set_ylim(0, 500)
        ax7.set_ylabel('Belt Scale 1\n'
                       ' [t]', rotation=90)
        ax7.set_yticks([0, 250, 500])
        ax7.legend(loc=4, prop={'size': 10})
        ax7.grid(alpha=0.5)
        # save fig
        plt.tight_layout()
        plt.savefig(fr'03_plots\BBT_S_{FROM}_{TO}.png', dpi=600)
        
plot(df, 22000, 24000, 35)

