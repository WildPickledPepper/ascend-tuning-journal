"""On-demand eager-prefill final-state export; disabled for all benchmarks."""
from pathlib import Path
import json,os
from graph_audit_base import GraphAudit as Base

class GraphAudit(Base):
    def prepare_capture(self, sample_id, token_ids, loss_mask, split):
        import torch
        assert split in ('train','validation') and isinstance(sample_id,int)
        assert len(token_ids)==len(loss_mask) and 4<=len(token_ids)<=512
        assert self.model_runner.vllm_config.cache_config.enable_prefix_caching is False
        self._capture_pending=(sample_id,token_ids,loss_mask,split)
        if not hasattr(self,'_capture_hook'):
            model=self.model_runner.get_model()
            def capture(module,args,kwargs,out):
                pending=getattr(self,'_capture_pending',None)
                if pending is None:return
                self._capture_pending=None
                sample_id,expected,mask,split=pending
                ids=kwargs.get('input_ids',args[0] if args else None)
                pos=kwargs.get('positions',args[1] if len(args)>1 else None)
                actual=ids.detach().cpu().tolist()
                assert actual==expected, 'Capture must see whole unpadded single prefill'
                positions=pos.detach().cpu()
                if positions.ndim==2:
                    assert torch.equal(positions[0],positions[1]) and torch.equal(positions[0],positions[2])
                    positions=positions[0]
                assert positions.tolist()==list(range(len(expected)))
                hidden=out[0] if isinstance(out,tuple) else out
                assert hidden.shape==(len(expected),5120) and hidden.dtype==torch.bfloat16
                logits=module.compute_logits(hidden[-1:])
                next_id=logits.argmax(-1).item()
                hidden_cpu=hidden.detach().cpu().contiguous()
                assert torch.isfinite(hidden_cpu).all()
                root=Path(os.environ['DFLASH_RUN'])/'features'/split;root.mkdir(parents=True,exist_ok=True)
                path=root/(str(sample_id)+'.ckpt')
                torch.save({'input_ids':torch.tensor(expected,dtype=torch.long),'loss_mask':torch.tensor(mask,dtype=torch.float32),'target_last_hidden_states':hidden_cpu},path)
                (root/(str(sample_id)+'.json')).write_text(json.dumps({'tokens':len(expected),'supervised':sum(mask),'next_token_argmax':next_id,'post_norm':True,'feature_dtype':'bfloat16','input_positions_verified':True}))
                self._capture_last={'sample_id':sample_id,'tokens':len(expected),'next_token_argmax':next_id}
            self._capture_hook=model.register_forward_hook(capture,with_kwargs=True)
        return {'armed':sample_id}

    def capture_result(self):
        assert self._capture_pending is None
        return self._capture_last
