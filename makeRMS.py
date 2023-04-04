import pandas as pd 
import matplotlib.pyplot as plt


df = pd.read_csv('RMS_MMS_masspoints.csv')

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots() 
for key in df:
    print(key)
    if 'rms2' in key:
        ax3.plot(df['masspoints'], df[key], label=key)
    elif 'sensitivity' in key:
        ax4.plot(df['masspoints'], df[key], label=key)
    elif 'rms' in key:
        ax1.plot(df['masspoints'], df[key], label=key)
    elif 'mms' in key:
        ax2.plot(df['masspoints'], df[key], label=key)
    

ax1.set_title(f'a1 RMS vs mass')
ax1.set_xlabel('Mass (GeV)')
ax1.set_ylabel('RMS')
ax1.legend()
fig1.savefig(f'plots/trend_RMS_masspoints.png', bbox_inches='tight')

ax2.set_title(f'a1 MMS vs mass')
ax2.set_xlabel('Mass (GeV)')
ax2.set_ylabel('MMS')
ax2.legend()
fig2.savefig(f'plots/trend_MMS_masspoints.png', bbox_inches='tight')

ax3.set_title(f'a1 RMS (no log2) vs mass')
ax3.set_xlabel('Mass (GeV)')
ax3.set_ylabel('RMS (no log2)')
ax3.legend()
fig3.savefig(f'plots/trend_RMS_nolog2_masspoints.png', bbox_inches='tight')

ax4.set_title(f'a1 sensitivity vs mass')
ax4.set_xlabel('Mass (GeV)')
ax4.set_ylabel('sensitivity')
ax4.legend()
fig4.savefig(f'plots/trend_sensitivity_masspoints.png', bbox_inches='tight')
    





