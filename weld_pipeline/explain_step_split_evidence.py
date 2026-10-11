"""Show actual annotation cuts and verified CAD identities, no heuristic rule."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from training.labels import sha256
from weld_pipeline.truth import load_truth


def main():
    root=Path('weld_pipeline/reports/step_alignment_20260930')
    rows=[];examples=[]
    for name in ['224-GR24A1','527-FR146A']:
        truth,_=load_truth(Path('DataSet/9月20日')/name)
        v=[w for w in truth if w['type']=='vertical']
        candidates=json.loads((root/(name+'_attribute_candidates.json')).read_text('utf8'))
        by_parent=defaultdict(list)
        for r in candidates['children']:
            if r['type']=='vertical':
                by_parent[r['weld_id']].append(r)
        consistent=0;ambiguities=[]
        for w in v:
            identities=[]
            for r in by_parent[w['id']]:
                sample=r['samples'][2]
                if sample['unique_pair']:
                    identities.append(tuple(sorted(p['volume'] for p in sample['candidate_pairs'][0])))
            if len(identities)==len(w['segments']) and len(set(identities))==1:
                consistent+=1
            else:
                ambiguities.append(w['id'])
        lengths=lambda w:[float(np.linalg.norm(np.diff(np.array(s['points']),axis=0),axis=1).sum()) for s in w['segments']]
        counts=Counter(round(sum(lengths(w)),3) for w in v)
        example=next(w for w in v if abs(sum(lengths(w))-120.)<.01)
        examples.append((name,example,lengths(example)))
        rows.append({'name':name,'vertical_parent_count':len(v),'vertical_child_count':sum(len(w['segments']) for w in v),
            'parent_lengths_mm':dict(counts),'child_counts':dict(Counter(len(w['segments']) for w in v)),
            'same_contact_board_identity_at_all_child_midpoints':consistent,'unresolved_parent_ids':ambiguities,
            'example_id':example['id'],'example_child_lengths_mm':lengths(example),
            'candidate_file_sha256':sha256(root/(name+'_attribute_candidates.json'))})
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,1,figsize=(9,4.8),sharex=True)
    for ax,(name,example,lengths) in zip(axes,examples):
        start=0.
        for i,length in enumerate(lengths):
            ax.barh(0,length,left=start,height=.35,color=['#536c87','#758ba3','#a5b4c3'][i],edgecolor='white')
            ax.text(start+length/2.,0,f'第{i+1}段\n{length:.1f} mm',ha='center',va='center',color='white',fontsize=11)
            ax.plot([start,start],[-.25,.25],color='black',lw=1.)
            start+=length
        ax.plot([start,start],[-.25,.25],color='black',lw=1.)
        ax.set_ylim(-.5,.5);ax.set_yticks([]);ax.set_title(f'{name}：原标注 120 mm / {len(lengths)} 子段',loc='left',fontsize=12)
        for spine in ['left','right','top']:
            ax.spines[spine].set_visible(False)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    axes[-1].set_xlim(0,120);axes[-1].set_xticks(np.arange(0,121,20));axes[-1].set_xlabel('沿原标注焊缝的距离 / mm')
    fig.text(.08,.03,'STEP核查：GR各子段中点仍接触同一对板件。三段为标注切分；长度相同不能决定切分。',fontsize=10)
    fig.subplots_adjust(left=.08,right=.97,top=.93,bottom=.19,hspace=.7)
    fig.savefig(root/'同长立焊_原标注分段对照.png',dpi=150);plt.close(fig)
    (root/'split_continuity_evidence.json').write_text(json.dumps({'rows':rows,'full_contract_pass':False,
        'scope':'Reference annotation and CAD contact identities; not a universal process rule.',
        'source_sha256':sha256(__file__)},ensure_ascii=False,indent=2),encoding='utf8')
    print('SPLIT_EVIDENCE',json.dumps(rows,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
