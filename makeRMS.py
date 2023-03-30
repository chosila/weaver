import pandas as pd 
import matplotlib.pyplot as plt

def makeRMSMMS(csvName, name_mod):
    df = pd.read_csv(csvName)
    
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for key in df:
        if 'rms' in key:
            ax1.plot(df['masspoints'], df[key], label=key)
        elif 'mms' in key:
            ax2.plot(df['masspoints'], df[key], label=key)
        
    
    ax1.set_title(f'a1 RMS ({name_mod}) vs mass')
    ax1.set_xlabel('Mass (GeV)')
    ax1.set_ylabel('RMS')
    ax1.legend()

    fig1.savefig(f'plots/RMS_{name_mod}_masspoints.png', bbox_inches='tight')
    ax2.set_title(f'a1 MMS ({name_mod}) vs mass')
    ax2.set_xlabel('Mass (GeV)')
    ax2.set_ylabel('MMS')
    ax2.legend()
    
    fig2.savefig(f'plots/MMS_{name_mod}_masspoints.png', bbox_inches='tight')



for csvName, name_mod in zip(['RMS_MMS_MsPnt.csv', 'RMS_MMS_MsPnt_log2.csv'], ['mass', 'log2(mass_ratio)']):
    makeRMSMMS(csvName, name_mod)
