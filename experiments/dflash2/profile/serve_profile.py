"""Install host profiling hooks in both API and spawned worker processes."""
from profile_adapter import install
install()
if __name__=='__main__':
    import runpy,vllm_ascend
    vllm_ascend.register_model()
    runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')
