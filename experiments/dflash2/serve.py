"""Register the installed Ascend model plugin before frontend draft validation."""
if __name__ == '__main__':
    import runpy
    import vllm_ascend

    vllm_ascend.register_model()
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')
