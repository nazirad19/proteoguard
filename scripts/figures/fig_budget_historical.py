"""HISTORICAL illustration from pasted values, NOT regenerated experimental results.

Use fig_budget.py --input anchor_budget.csv --output figures/f7.png instead.
This file is retained solely to trace the figure in the original draft.
"""
from pathlib import Path
Path("paperfigs").mkdir(exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
K=[0,1,3,5,10,20]
# from cell AB output (mean over two data splits); multi at K=0,1 equals shared
d={("totalVI","shared"):dict(r2=[0,.860,.849,.862,.853,.798],b=[.075,.051,.051,.050,.049,.047],e=[.071,.026,.028,.026,.025,.030],c=[.087,.296,.292,.297,.296,.285]),
   ("totalVI","multi"): dict(r2=[0,.860,.881,.888,.889,.906],b=[.075,.051,.050,.048,.048,.047],e=[.071,.026,.025,.024,.024,.024],c=[.087,.296,.304,.310,.316,.309]),
   ("ridge","shared"):  dict(r2=[0,.851,.865,.866,.871,.845],b=[.085,.043,.042,.040,.042,.040],e=[.085,.026,.022,.023,.023,.025],c=[.088,.262,.271,.270,.273,.267]),
   ("ridge","multi"):   dict(r2=[0,.851,.869,.884,.890,.913],b=[.085,.043,.043,.040,.040,.038],e=[.085,.026,.023,.022,.026,.021],c=[.088,.262,.270,.280,.286,.288])}
fam={"activation":[0,.824,.873,.874,.866,.908],"exhaustion/inhibitory":[0,.898,.906,.915,.914,.924],
     "lineage":[0,.609,.660,.659,.753,.799],"trafficking":[0,.851,.866,.878,.862,.900]}
col={("totalVI","shared"):"#1f6fe5",("totalVI","multi"):"#ff8c00",("ridge","shared"):"#18a558",("ridge","multi"):"#8e44ad"}
fcol=["#1f6fe5","#ff8c00","#18a558","#8e44ad"]
x=np.arange(len(K))
fig,ax=plt.subplots(1,4,figsize=(17,4.3))
for key,v in d.items():
    lab=f"{key[0]}, {key[1]} map"
    ax[0].plot(x,v["r2"],"o-",color=col[key],label=lab,lw=2)
    ax[1].plot(x,np.array(v["b"])*100,"o-",color=col[key],lw=2)
    ax[2].plot(x,np.array(v["e"])*100,"o-",color=col[key],lw=2)
for (f,v),c in zip(fam.items(),fcol): ax[3].plot(x,v,"o-",color=c,label=f,lw=2)
titles=["Offset R² on new participants (median over proteins)","Participant–protein pairs below 85% coverage (%)",
        "Error among biomarker calls (%)","Offset R² by protein family (totalVI, multi map)"]
for a,t in zip(ax,titles):
    a.set_xticks(x); a.set_xticklabels(K); a.set_xlabel("Number of measured sentinel proteins"); a.set_title(t,fontsize=10); a.grid(alpha=.3)
ax[0].set_ylim(-.02,1); ax[3].set_ylim(-.02,1)
h,l=ax[0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",ncol=4,bbox_to_anchor=(0.38,-0.04),frameon=False)
ax[3].legend(loc="lower right",fontsize=8,frameon=False)
plt.tight_layout(rect=(0,0.06,1,1)); plt.savefig("paperfigs/f7.png",dpi=160,bbox_inches="tight")
