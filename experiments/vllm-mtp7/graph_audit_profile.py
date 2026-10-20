"""Read-only worker inspection, called outside the timed benchmark."""


class GraphAudit:
    def enable_profile_scopes(self):
        import torch
        from vllm.v1 import utils
        previous = getattr(utils._PROFILER_FUNC, '__name__', None)
        utils._PROFILER_FUNC = torch.profiler.record_function
        return {'previous_scope_function': previous, 'scope_function': 'record_function',
                'scope': 'Host labels enabled only after unprofiled benchmark completed.'}

    def graph_audit(self):
        from vllm_ascend.compilation.acl_graph import _acl_graph_wrappers

        runner = self.model_runner
        drafter = runner.drafter
        runnable = drafter._runnable
        wrappers = []
        for wrapper in list(_acl_graph_wrappers):
            entries = []
            for key, entry in wrapper.concrete_aclgraph_entries.items():
                entries.append({"batch_descriptor": str(key),
                                "captured": entry.aclgraph is not None})
            fn = wrapper.runnable
            wrappers.append({"is_draft_runnable": wrapper is runnable,
                             "callable": getattr(fn, '__qualname__', type(fn).__name__),
                             "runtime_mode": str(wrapper.runtime_mode), "entries": entries})
        cfg = runner.vllm_config
        return {"num_speculative_tokens": drafter.num_speculative_tokens,
                "draft_use_cuda_graph": drafter.use_cuda_graph,
                "draft_enforce_eager": cfg.speculative_config.enforce_eager,
                "main_graph_mode": str(cfg.compilation_config.cudagraph_mode),
                "main_capture_sizes": cfg.compilation_config.cudagraph_capture_sizes,
                "dtype": str(cfg.model_config.dtype),
                "kv_cache_dtype": str(cfg.cache_config.cache_dtype),
                "mamba_cache_dtype": str(cfg.cache_config.mamba_cache_dtype),
                "mamba_ssm_cache_dtype": str(cfg.cache_config.mamba_ssm_cache_dtype),
                "tp_size": cfg.parallel_config.tensor_parallel_size,
                "context_length": cfg.model_config.max_model_len,
                "max_num_seqs": cfg.scheduler_config.max_num_seqs,
                "prefix_caching": cfg.cache_config.enable_prefix_caching,
                "gpu_memory_utilization": cfg.cache_config.gpu_memory_utilization,
                "wrappers": wrappers}
