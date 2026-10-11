from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import FancyBboxPatch

from action_selection_scoring import confusion_table

def generate(benchmark, primary, cooperative, action_metrics, action, destination):
    F=Path(destination)
    F.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':11,'savefig.dpi':300,'pdf.fonttype':42})
    models=['Qwen2.5-7B-Instruct','Gemma-4-E4B-it','Mistral-7B-Instruct-v0.2']
    short={'Qwen2.5-7B-Instruct':'Qwen','Gemma-4-E4B-it':'Gemma','Mistral-7B-Instruct-v0.2':'Mistral','Heuristic-Baseline':'Heuristic','Always-Clarify-Baseline':'Always clarify'}
    colors=['#2878B5','#E58B2A','#41935B','#858585']
    def save(fig,name):
        fig.savefig(F/(name+'.png'),bbox_inches='tight',facecolor='white', metadata={'Software':'AmbiTab-Sports'})
        with Image.open(F/(name+'.png')) as pixels:
            pixels.load()
            pixels.info.clear()
            pixels.save(F/(name+'.png'))
        fig.savefig(F/(name+'.pdf'),bbox_inches='tight',facecolor='white', metadata={'CreationDate':None,'ModDate':None,'Creator':'AmbiTab-Sports'})
        plt.close(fig)

    fig,ax=plt.subplots(figsize=(10.5,5.6));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    def box(x,y,w,h,s,edge='#495057',fs=11):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01',lw=1.5,edgecolor=edge,facecolor='white'))
        ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=fs,linespacing=1.5)
    box(.025,.385,.2,.23,'Sports-data\nquery\n(no table evidence)',fs=12)
    ys=[.69,.39,.09]
    labels=['Unambiguous (U)\nANSWER\nRecorded correctness','Cooperative ambiguous (C)\nINTERPRET\nRecorded three-dimension rubric','Uncooperative (X)\nCLARIFY\nAppropriate clarification']
    for y,label in zip(ys,labels):
        box(.335,y,.35,.22,label)
        ax.annotate('',xy=(.325,y+.11),xytext=(.235,.5),arrowprops={'arrowstyle':'->','lw':1.4})
        ax.annotate('',xy=(.765,.5),xytext=(.695,y+.11),arrowprops={'arrowstyle':'->','lw':1.4})
    box(.775,.26,.2,.48,'Unified action selection\non all 360 queries\n\nSame prompt;\ntrack label hidden\n\nPre-retrieval scope',fs=11)
    save(fig,'Fig1_Framework')

    domains=['Club\nfootball','International\nfootball','Olympics','Basketball','Tennis','Formula 1']
    fig,ax=plt.subplots(figsize=(9,4.4));x=np.arange(6)
    for i,(label,color) in enumerate(zip(['Unambiguous','Cooperative ambiguous','Uncooperative'],colors)):
        ax.bar(x+(i-1)*.25,[20]*6,.23,color=color,label=label)
    ax.set_xticks(x,domains);ax.set_ylim(0,25);ax.set_yticks([0,5,10,15,20]);ax.set_ylabel('Queries per track and subdomain')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.19),ncol=3,frameon=False)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout()
    save(fig,'Fig2_Composition')

    e=primary.set_index('model').loc[models]
    fig,ax=plt.subplots(figsize=(7.5,4.4));v=e.accuracy.to_numpy()
    ax.bar(np.arange(3),v,color=colors[:3],width=.58)
    ax.errorbar(np.arange(3),v,yerr=np.vstack([v-e.ci_low.to_numpy(),e.ci_high.to_numpy()-v]),fmt='none',color='black',capsize=5,lw=1.4)
    for i,r in enumerate(e.itertuples()):ax.text(i,r.ci_high+.02,f'{r.correct}/120\n{r.accuracy:.3f}',ha='center',fontsize=11)
    ax.set_xticks(np.arange(3),[short[m] for m in models]);ax.set_ylim(0,.53);ax.set_ylabel('Recorded binary correctness');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();save(fig,'Fig3_Execution')

    c=cooperative.set_index('model').loc[models+['Heuristic-Baseline']]
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),gridspec_kw={'width_ratios':[1,1.25]})
    ax=axes[0];v=c.human_cooperative_score.to_numpy();x=np.arange(4)
    ax.bar(x,v,color=colors,width=.6);ax.errorbar(x,v,yerr=np.vstack([v-c.ci_low.to_numpy(),c.ci_high.to_numpy()-v]),fmt='none',color='black',capsize=4,lw=1.2)
    for i,r in enumerate(c.itertuples()):ax.text(i,r.ci_high+.035,f'{r.human_cooperative_score:.3f}',ha='center',fontsize=10)
    ax.set_xticks(x,[short[m] for m in c.index],rotation=20,ha='right');ax.set_ylim(0,1);ax.set_ylabel('Normalized adjudicated score');ax.set_title('(a) Total score and 95% CI')
    ax=axes[1];x=np.arange(3)
    for i,m in enumerate(c.index):
        vals=c.loc[m,['interpretation_validity_norm','assumption_quality_norm','answer_usefulness_norm']].to_numpy(float)
        ax.bar(x+(i-1.5)*.19,vals,.18,color=colors[i],label=short[m])
    ax.set_xticks(x,['Interpretation\nvalidity','Assumption\nquality','Answer\nusefulness']);ax.set_ylim(0,1);ax.set_ylabel('Dimension score / 2');ax.set_title('(b) Rubric dimensions')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.0),ncol=2,frameon=False,fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout(w_pad=2);save(fig,'Fig4_Cooperative')

    cm=confusion_table(action);actions=['ANSWER','INTERPRET','CLARIFY']
    fig,axes=plt.subplots(2,2,figsize=(9.8,8.2),layout='constrained')
    for ax,m in zip(axes.flat,models+['Always-Clarify-Baseline']):
        d=cm[cm.model==m].pivot(index='expected_action',columns='predicted_action',values='row_proportion').reindex(index=actions,columns=actions)
        arr=d.to_numpy();im=ax.imshow(arr,vmin=0,vmax=1,cmap='Blues')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{arr[i,j]:.3f}',ha='center',va='center',color='white' if arr[i,j]>.6 else '#17212b',fontsize=12)
        ax.set_xticks(range(3),['Answer','Interpret','Clarify']);ax.set_yticks(range(3),['Answer','Interpret','Clarify']);ax.set_xlabel('Predicted action');ax.set_ylabel('Expected action');ax.set_title(short[m],weight='bold')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.85,pad=.025,label='Proportion within expected action')
    save(fig,'Fig5_Action_Confusion')

    b=action_metrics.set_index('model')
    master=pd.read_csv(F.parent/'results/FINAL_master_results_steps1_3.csv').set_index('model')
    cols=['execution_accuracy','cooperative_human_score','action_accuracy','appropriate_clarification_X','over_clarification_UC','unsafe_answer_or_interpret_X']
    labs=['Execution\ncorrectness','Cooperative\nrecorded score','Action\naccuracy','Appropriate\nclarification','Over-\nclarification*','Non-clarification\non X*']
    fig,ax=plt.subplots(figsize=(11,4.8));x=np.arange(6)
    for i,m in enumerate(models):ax.bar(x+(i-1)*.24,master.loc[m,cols].to_numpy(float),.23,color=colors[i],label=short[m])
    ax.set_xticks(x,labs);ax.set_ylim(0,1.12);ax.set_yticks(np.arange(0,1.01,.2));ax.set_ylabel('Proportion or normalized score');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.17),ncol=3,frameon=False)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.08,.015,'* Lower is preferable. Different dimensions have different meanings; this is not an aggregate ranking.',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);save(fig,'Fig6_Behavior_Profiles')
