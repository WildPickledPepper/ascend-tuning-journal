"""Use the stock model server with an optional, inactive-until-started profiler."""
from profile_factory import install
install()
if __name__ == '__main__':
    import runpy
    import vllm_ascend
    vllm_ascend.register_model()
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')
