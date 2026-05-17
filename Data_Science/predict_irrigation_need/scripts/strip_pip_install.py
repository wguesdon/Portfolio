#!/usr/bin/env python
"""Strip pip install blocks from cuML scripts and copy to aws/ for Docker build."""
import re
from pathlib import Path

BASE = Path("/mnt/data/Github/Kaggle/Playground_Series/PS6E4")

for script in ["train_rf_ote_cuml.py", "train_svm_ote_cuml.py"]:
    content = (BASE / "scripts" / script).read_text()

    # Remove subprocess and sys imports (only needed for pip install)
    content = content.replace("import subprocess\n", "")
    content = content.replace("import sys\n", "")

    # Remove the install blocks
    content = re.sub(
        r"# Install cuML.*?stdout=subprocess\.DEVNULL,\n\)\n",
        "",
        content,
        flags=re.DOTALL,
    )

    (BASE / "aws" / script).write_text(content)
    print(f"Created aws/{script}")
