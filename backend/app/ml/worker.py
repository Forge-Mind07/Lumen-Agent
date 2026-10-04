"""Private subprocess entry point. No user Python/code execution is supported."""
from __future__ import annotations

import json
import math
import sys
import os
from pathlib import Path


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(job_path: str):
    if sys.platform == "linux":
        # A server crash must not leave a detached training process running.
        import ctypes
        import signal
        parent = os.getppid()
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        if parent == 1 or os.getppid() != parent:
            return
    from .datasets import load_frame
    from .inference import infer
    from .patterns import forecast, patterns
    from .preprocessing import inspect
    from .supervised import train
    from threadpoolctl import threadpool_limits
    job = json.loads(Path(job_path).read_text())
    directory = Path(job_path).parent
    def progress(stage, label):
        print(json.dumps({"stage": stage, "label": label, "status": "running"}), flush=True)
    try:
        import pandas as pd
        progress("preparing", "Preparing dataset")
        frame = pd.DataFrame([job["arguments"]["row"]]) if job["arguments"].get("row") is not None else load_frame(job["dataset"]["path"])
        with threadpool_limits(limits=1):
            operation, args = job["operation"], job["arguments"]
            if operation == "inspect_dataset":
                result = inspect(frame, args.get("target"))
            elif operation == "train_model":
                result = train(frame, args, directory, progress)
            elif operation == "cluster_dataset":
                result = patterns(frame, args, directory, progress, "clustering")
            elif operation == "detect_anomalies":
                result = patterns(frame, args, directory, progress, "anomaly_detection")
            elif operation == "forecast_series":
                result = forecast(frame, args, directory, progress)
            elif operation in {"predict", "evaluate_model", "explain_prediction"}:
                result = infer(frame, args, job["model"], directory, operation)
            else:
                raise ValueError("Unsupported runtime operation")
        import sklearn
        import numpy as np
        result["runtime_versions"] = {"python": sys.version.split()[0], "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}
        (directory / "result.json").write_text(json.dumps({"ok": True, "result": clean(result)}, allow_nan=False))
    except Exception as exc:
        message = str(exc)[:600] if isinstance(exc, ValueError) else "The ML runtime could not complete this operation. Review the dataset and configuration."
        (directory / "result.json").write_text(json.dumps({"ok": False, "error": message}))


if __name__ == "__main__":
    run(sys.argv[1])
