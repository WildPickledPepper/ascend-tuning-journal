"""Reconstruct the isolated capture wrapper; no kernel or installed-source edit.

Usage: python prepare.py PRIOR_PACKED_PREFILL_RUN NEW_CAPTURE_RUN
Keep all deployment paths in the private environment. Run capture.py with the
same DEPLOY_ENV/SERVE_PYTHON/QWEN_MODEL_PATH/GSM8K_TEST_PATH and three library
variables documented by the prior packed-prefill experiment.
"""
import json
from pathlib import Path
import shutil
import sys

source, target = [Path(x).resolve() for x in sys.argv[1:3]]
target.mkdir(exist_ok=False)
(target/"results").mkdir()
(target/"profiles").mkdir()
for name in ["start.sh","serve.py","client.py","model_extension.py","graph_audit_base.py"]:
    shutil.copy2(source/name,target/name)
for name in ["profile_factory.py","capture.py","analyze.py"]:
    shutil.copy2(Path(__file__).parent/name,target/name)
if (source/"cache-1").exists():
    (target/"cache-1").symlink_to(source/"cache-1",target_is_directory=True)
path = target/"serve.py"
path.write_text("from profile_factory import install\ninstall()\n"+path.read_text())
path = target/"model_extension.py"
path.write_text(path.read_text()+"\nfrom profile_factory import ProfileControl\nOriginalGraphAudit = GraphAudit\nclass GraphAudit(ProfileControl, OriginalGraphAudit):\n    pass\n")
config = json.dumps({"profiler":"torch","torch_profiler_dir":str(target/"profiles"),
                     "torch_profiler_with_stack":False,"torch_profiler_with_memory":False,"ignore_frontend":True})
path = target/"start.sh"
path.write_text(path.read_text().replace("--no-enable-log-requests",
                "--no-enable-log-requests \\\n --profiler-config '"+config+"'"))
print("Prepared isolated capture directory:",target)
