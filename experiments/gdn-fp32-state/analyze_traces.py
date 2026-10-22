"""Align target predictions by exact text prefix, not by draft round number."""
import array
import hashlib
import json
from pathlib import Path
import sys

import torch


def digest(ids):
    return hashlib.sha256(array.array('i',ids).tobytes()).hexdigest()


def load(root,label):
    trace=torch.load(root/(label+'-trace.pt'),map_location='cpu',weights_only=True)
    response=json.loads((root/(label+'-request.json')).read_text())
    tokens=response['token_ids']
    prompt=trace['records'][0]['input_ids'][:trace['records'][0]['prompt_length']]
    aligned={}
    mismatches=[]
    for ri,record in enumerate(trace['records']):
        for r,row in enumerate(record['rows']):
            j=row['output_index']
            if 0<=j<=len(tokens) and row['prefix_sha256']==digest(prompt+tokens[:j]):
                item=dict(row,hidden=record['hidden'][r],round_index=ri,row_index=r)
                if r in record['full_logits']:
                    item['full_logits']=record['full_logits'][r]
                aligned[j]=item
        start=record['rows'][0]['output_index']
        sampled=record['sampled_ids'][0]
        for i,t in enumerate(sampled):
            if t<0:
                continue
            if i<len(record['rows']) and t!=record['rows'][i]['argmax_id']:
                mismatches.append({'round':ri,'row':i,'kind':'sample_not_target_argmax','sample':t,
                                   'argmax':record['rows'][i]['argmax_id']})
            if start+i<len(tokens) and t!=tokens[start+i]:
                mismatches.append({'round':ri,'row':i,'kind':'stream_not_sample','sample':t,'stream':tokens[start+i]})
    return tokens,prompt,aligned,{'tokens':len(tokens),'rounds_including_prefill':len(trace['records']),
                                'aligned_target_positions':len(aligned),'sampling_mismatches':mismatches}


def compare(left,right):
    lt,lp,la,ls=left
    rt,rp,ra,rs=right
    assert lp==rp
    first=next((j for j,(x,y) in enumerate(zip(lt,rt)) if x!=y),None)
    bound=first if first is not None else min(len(lt),len(rt))-1
    hidden_diffs=[]
    top_diffs=[]
    for j in sorted(set(la)&set(ra)):
        if j>bound:
            continue
        a,b=la[j],ra[j]
        assert a['prefix_sha256']==b['prefix_sha256']
        e=(a['hidden'].float()-b['hidden'].float()).abs()
        n=int((a['hidden']!=b['hidden']).sum())
        if n:
            hidden_diffs.append({'position':j,'different':n,'max_abs':float(e.max()),
                                 'left_round':a['round_index'],'left_row':a['row_index'],
                                 'right_round':b['round_index'],'right_row':b['row_index']})
        if a['argmax_id']!=b['argmax_id']:
            top_diffs.append(j)
    detail=None
    if first is not None and first in la and first in ra:
        a,b=la[first],ra[first]
        detail={
            'same_prefix':a['prefix_sha256']==b['prefix_sha256'],
            'left_output_id':lt[first], 'right_output_id':rt[first],
            'left_argmax':a['argmax_id'],'right_argmax':b['argmax_id'],
            'left_top8_ids':a['top8_ids'],'left_top8_values':a['top8_values'],
            'right_top8_ids':b['top8_ids'],'right_top8_values':b['top8_values'],
            'left_round':a['round_index'],'left_row':a['row_index'],
            'right_round':b['round_index'],'right_row':b['row_index'],
        }
        if 'full_logits' in a and 'full_logits' in b:
            av,bv=a['full_logits'].float(),b['full_logits'].float()
            detail.update(full_logits_different=int((av!=bv).sum()),full_logits_max_abs=float((av-bv).abs().max()))
            ids=set(a['top8_ids'][:2]+b['top8_ids'][:2])
            detail['paired_candidate_scores']={str(i):[float(av[i]),float(bv[i])] for i in ids}
    return {'left_tokens':len(lt),'right_tokens':len(rt),'tokens_equal':lt==rt,
            'first_different_output_index':first,'hidden_difference_count_before_output_split':len(hidden_diffs),
            'first_hidden_difference':hidden_diffs[0] if hidden_diffs else None,
            'last_hidden_difference_before_split':hidden_diffs[-1] if hidden_diffs else None,
            'target_argmax_difference_positions_before_split':top_diffs,'split_detail':detail}


def main(root):
    labels=['full-a','masked','full-b']
    arms={label:load(root,label) for label in labels}
    result={'arms':{label:x[3] for label,x in arms.items()},'comparisons':{}}
    for a,b in [('full-a','full-b'),('full-a','masked')]:
        result['comparisons'][a+'__'+b]=compare(arms[a],arms[b])
    (root/'trace-analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main(Path(sys.argv[1]))
