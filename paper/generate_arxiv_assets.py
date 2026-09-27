"""Regenerate arXiv revision figures and tables from immutable saved results."""
from pathlib import Path
import ast
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results'
FIG = ROOT / 'paper/figures'
SEC = ROOT / 'paper/sections'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
})

def read(path):
    return json.loads((RES / path).read_text(encoding='utf-8'))

def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(FIG / f'{name}.{ext}', bbox_inches='tight', pad_inches=.05)
    plt.close(fig)

def table(name, caption, label, columns, heading, rows):
    text = '\n'.join([
        r'\begin{table}[htbp]', r'\centering', r'\caption{' + caption + '}',
        r'\label{' + label + '}', r'\small',
        r'\begin{tabular}{' + columns + '}', r'\toprule',
        heading + r' \\', r'\midrule',
        *[r'\midrule' if row is None else ' & '.join(row) + r' \\' for row in rows],
        r'\bottomrule', r'\end{tabular}', r'\end{table}', ''
    ])
    (SEC / (name + '.tex')).write_text(text, encoding='utf-8')

def pm(values, scale=100):
    a = np.array(values) * scale
    return rf'${a.mean():.2f}\pm{a.std(ddof=1):.2f}$'

def stat(v, scale=100):
    return rf'${v["mean"]*scale:.2f}\pm{v["sample_std"]*scale:.2f}$'

def weight_profiles():
    profiles = {ast.literal_eval(k): np.array(v) for k,v in
                read('d1_cross_system/weight_profiles.json').items()}
    fig, ax = plt.subplots(figsize=(4.7, 2.4), layout='constrained')
    for rho,color,marker in [(1.,'#2166ac','o'),(1.3,'#d17a20','s'),(1.5,'#b63c4b','^')]:
        w = profiles[('complex_modrelu',rho,1.2,10)]
        assert len(w)==40 and abs(w.sum()-1)<.01
        ax.plot(np.arange(1,41),w,color=color,marker=marker,ms=2.8,lw=1.5,
                markevery=2,label=rf'$\rho={rho:.1f}$')
    ax.axhline(1/40,color='.5',ls='--',lw=1,label='Uniform ($1/T$)')
    ax.set(xlabel='Trajectory timestep $t$',ylabel=r'GRACE weight $w_t$',xlim=(.5,40.5))
    ax.grid(axis='y',alpha=.2)
    ax.legend(loc='upper right',frameon=False)
    save(fig,'fig3_grace_method')

def modern_figures(panel):
    conds=panel['conditions']
    horizons=[8,16,24,32]
    colors=['#bd434a','#2675a2','#7551a5','#298c70']
    fig, axes=plt.subplots(1,2,figsize=(7.1,2.8),layout='constrained',gridspec_kw={'width_ratios':[1.22,1]})
    ax=axes[0]
    for c,color in zip(conds,colors):
        y=[c['terminal_by_horizon'][str(t)]['mean']*100 for t in horizons]
        err=[c['terminal_by_horizon'][str(t)]['sample_std']*100 for t in horizons]
        family='SwiGLU' if c['model']=='Recursive SwiGLU' else 'Attention'
        ax.errorbar(horizons,y,yerr=err,color=color,lw=1.5,marker='o',ms=3,
                    capsize=2,ls='-' if c['training_horizon']==8 else '--',
                    label=family+rf', $T_{{\rm train}}={c["training_horizon"]}$')
    ax.set(xlabel=r'Inference horizon $T_{\rm test}$',ylabel='Validation accuracy (%)',
           xticks=horizons,ylim=(74,82))
    ax.set_title('(a) Horizon extension',loc='left',fontsize=9,fontweight='bold')
    ax.grid(axis='y',alpha=.2)
    ax.legend(loc='lower left',frameon=False,fontsize=6.5)
    ax=axes[1]; x=np.arange(4); width=.35
    for key,offset,color,label in [('terminal_t32',-width/2,'#385f7f','Terminal'),('early_t32',width/2,'#ed9564','Early-8')]:
        ax.bar(x+offset,[100*c[key]['mean'] for c in conds],width,
               yerr=[100*c[key]['sample_std'] for c in conds],color=color,
               error_kw={'elinewidth':1,'capsize':2},label=label)
    ax.set(ylim=(74,82),xticks=x,xticklabels=['SwiGLU\n8','SwiGLU\n32','Attention\n8','Attention\n32'],
           xlabel=r'Architecture / $T_{\rm train}$')
    ax.set_title(r'(b) Readout at $T_{\rm test}=32$',loc='left',fontsize=9,fontweight='bold')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='upper right',frameon=False)
    save(fig,'fig_modern_horizon_readout')
    fig,ax=plt.subplots(figsize=(5.8,2.6),layout='constrained')
    ax.bar(x,[c['drop_pp']['mean'] for c in conds],yerr=[c['drop_pp']['sample_std'] for c in conds],
           color=colors,width=.62,error_kw={'elinewidth':1,'capsize':3},alpha=.8)
    for i,c in enumerate(conds):ax.scatter(i+np.array([-.12,0,.12]),c['drop_pp']['values'],s=18,color='black',zorder=3)
    ax.axhline(0,color='.2',lw=.8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.set(xticks=x,xticklabels=['SwiGLU / 8','SwiGLU / 32','Attention / 8','Attention / 32'],
           ylabel='Terminal $A(8)-A(32)$ (pp)',xlabel=r'Architecture / $T_{\rm train}$',ylim=(-1,7))
    save(fig,'fig_modern_horizon_drop')

def modern_tables(panel):
    rows=[]
    for c in panel['conditions']:
        rows.append([c['model'].replace('Recursive ',''),str(c['training_horizon']),
                     stat(c['terminal_t8']),stat(c['terminal_t32']),stat(c['early_t32']),stat(c['drop_pp'],1)])
    table('generated_modern_summary',r'\textbf{Final four-condition panel.} Accuracy is in percent; loss is $A(8)-A(32)$ in percentage points. Means and sample standard deviations across three seeds.',
          'tab:modern_summary','lrrrrr',r'Model & $T_{\rm train}$ & Terminal 8 & Terminal 32 & Early 8 & Loss (pp)',rows)
    rows=[]
    for horizon,dirname in [(8,'attention_t8'),(32,'attention_t32_control')]:
        entries=[read(f'modern_iterative_panel/{dirname}/seed_{seed}/readout_metrics.json') for seed in [101,102,103]]
        for t in [8,16,24,32]:
            rows.append([str(horizon),str(t)]+[pm([e['by_horizon'][str(t)]['accuracy'][k] for e in entries]) for k in ['terminal','early','uniform','adaptive','grace']])
        if horizon==8:rows.append(None)
    table('generated_attention_readouts',r'\textbf{All frozen attention readouts.} Validation accuracy (\%, mean $\pm$ sample standard deviation, three seeds).',
          'tab:attention_readouts','rrrrrrr',r'$T_{\rm train}$ & $T_{\rm test}$ & Terminal & Early 8 & Uniform & Adaptive & \grace{}',rows)
    rows=[]; probe=read('modern_iterative_panel/followup_analysis/audits/e16_probe_audit.json')
    for horizon in [8,32]:
        e=probe['by_training_horizon'][str(horizon)]
        rows.append(['SwiGLU',str(horizon)]+[stat(e[k]) for k in ['terminal_step8','early_mean_steps1_to_8','terminal_step32']]+[stat(e['probe_drop_pp'],1)])
    for horizon,dirname in [(8,'attention_t8'),(32,'attention_t32_control')]:
        entries=[read(f'modern_iterative_panel/{dirname}/seed_{seed}/probe_metrics.json') for seed in [101,102,103]]
        rows.append(['Attention',str(horizon)]+[pm([e['accuracy'][k] for e in entries]) for k in ['terminal_step8','early_mean_steps1_to_8','terminal_step32']]+[pm([e['terminal_step8_minus_step32_pp'] for e in entries],1)])
    table('generated_modern_probes',r'\textbf{Fixed-alpha probe accuracy.} Separately fitted probes share ridge $\alpha=1$ and a fixed training subset. Accuracy in percent; loss in pp.',
          'tab:modern_probes','lrrrrr',r'Model & $T_{\rm train}$ & Step 8 & Early 8 & Step 32 & Loss (pp)',rows)
    audit=read('modern_iterative_panel/followup_analysis/audits/checkpoint_sensitivity.json');rows=[]
    for e in audit['rows']:
        role={'selected':'Selected','final_epoch':'Final epoch','saved_rank_2':'Saved rank 2','saved_rank_3':'Saved rank 3'}[e['checkpoint_role']]
        rows.append(['SwiGLU' if e['model']=='e16_t8' else 'Attention',str(e['seed']),role,str(e['epoch']),
                    f"{100*e['terminal_t8']:.2f}",f"{100*e['terminal_t32']:.2f}",f"{e['drop_pp']:.2f}"])
    table('generated_checkpoint_table',r'\textbf{Checkpoint sensitivity of eight-step-trained models.} Accuracy in percent; loss in pp. Only retained checkpoints are compared.',
          'tab:checkpoint_sensitivity','lr l rrrr',r'Model & Seed & Checkpoint & Epoch & Step 8 & Step 32 & Loss',rows)
    rows=[]
    for horizon,dirname in [(8,'rebuttal_recursive_tinyimagenet'),(32,'rebuttal_recursive_tinyimagenet_aligned_t32')]:
        data=read(f'{dirname}/seed_0_metrics.json')['shared_head']
        for gain,e in sorted(data.items(),key=lambda x:float(x[0])):
            # Shared-head metrics contain the frozen readout accuracies.
            v=e['readout_accuracy'] if 'readout_accuracy' in e else e['accuracy']
            rows.append([str(horizon),gain]+[f'{100*v[k]:.2f}' for k in ['terminal','early','uniform','adaptive','grace','fixed_profile']])
        if horizon==8:rows.append(None)
    table('generated_pilot_readouts',r'\textbf{Seed-0 SwiGLU pilot readouts at $T_{\rm test}=32$.} Accuracy in percent. Fixed denotes the training-derived global \grace{} profile. Neither pilot continued to further seeds.',
          'tab:pilot_readouts','rrrrrrrr',r'$T_{\rm train}$ & Gain & Terminal & Early 8 & Uniform & Adaptive & \grace{} & Fixed',rows)
    cost=read('rebuttal_recursive_tinyimagenet/runtime_memory.json');rows=[]
    for k,label in [('terminal','Terminal'),('early','Early window'),('adaptive','Adaptive'),('uniform','Uniform streaming'),('grace_naive',r'\grace{}, stored'),('grace_streaming',r'\grace{}, streaming')]:
        e=cost['rows'][k]
        rows.append([label,f"{e['full_inference']['median_ms']:.3f}",f"{e['readout_only']['median_ms']:.3f}",
                    f"{e['full_inference']['peak_additional_bytes']/2**20:.2f}",f"{e['readout_only']['peak_additional_bytes']/2**20:.2f}"])
    table('generated_cost_detail',r'\textbf{Full recurrent-head and readout-only benchmark.} Latencies in ms and additional peak allocations in MiB; the precomputed trajectory is part of the baseline in readout-only measurements.',
          'tab:cost_detail','lrrrr',r'Readout & Full ms & Readout ms & Full MiB & Readout MiB',rows)

if __name__=='__main__':
    panel=read('modern_iterative_panel/final_architecture_panel.json')
    weight_profiles();modern_figures(panel);modern_tables(panel)
    print('Regenerated revision figures and appendix tables from saved results.')
