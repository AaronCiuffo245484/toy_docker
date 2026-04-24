# ml-toy-repo: Learning Log

## Purpose

This repo exists to validate cross-platform TensorFlow Docker patterns
before transferring them to thalianacv.

---

## Finding 1: Use uv instead of Poetry for cross-platform TF projects

Poetry's lockfile resolves for the current platform only. A lockfile generated
on Mac ARM will not resolve correctly on Linux. uv handles platform markers
natively and resolves each platform independently via `[tool.uv] environments`.

**Pattern:**

```toml
[tool.uv]
environments = [
    "sys_platform == 'linux'",
    "sys_platform == 'darwin'",
]
```

---

## Finding 2: TensorFlow version mismatch between Mac and Linux is unavoidable

`tensorflow-macos==2.13.0` is the latest version supported by
`tensorflow-metal==1.1.0` on Mac ARM. The standard `tensorflow` package on
Linux can use a newer version. We use `2.15.0` on Linux.

**Pattern:**

```toml
"tensorflow==2.15.0; sys_platform == 'linux'",
"tensorflow-macos==2.13.0; sys_platform == 'darwin'",
"tensorflow-metal==1.1.0; sys_platform == 'darwin'",
```

---

## Finding 3: FastAPI version must be pinned to 0.100.0

`tensorflow-macos==2.13.0` pins `typing-extensions<4.6.0`. Modern FastAPI
requires `typing-extensions>=4.8.0`. These are irreconcilable. FastAPI 0.100.0
is the last version compatible with the TF 2.13 constraint. It supports all
features we need including `/docs`.

---

## Finding 4: Use src/ layout with hatchling

Same pattern as thalianacv. Package lives in `src/ml_toy_repo/`.
Hatchling must be told explicitly where to find it.

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/ml_toy_repo"]
```

---

## Finding 5: Model download endpoint should check file existence, not job state

Gating the `/model` endpoint on `job_state.status == COMPLETE` causes 400
errors after a server restart even when a valid model file exists on disk.
The correct pattern is to check whether the file exists and return 404 if not.

**Pattern:**

```python
@app.get("/model")
def download_model():
    if not Path(MODEL_PATH).exists():
        raise HTTPException(status_code=404, detail="No model file found. Run /train first.")
    return FileResponse(
        path=MODEL_PATH,
        media_type="application/octet-stream",
        filename="mnist.h5",
    )
```

---

## Finding 6: Use .h5 format for model saving

thalianacv uses `.h5` models. The `.h5` format is stable across TF 2.13 and
2.15 and should be used as the standard save format across both the toy repo
and thalianacv.

**Pattern:**

```python
model.save(MODEL_PATH, save_format="h5")
```

---

## Verified on Mac ARM (Apple M-series)

- TF version: 2.13.0
- Device: Apple M2 Metal GPU
- Training: 3 epochs, MNIST, minimal CNN, ~31 seconds
- Final accuracy: 0.38 (expected — minimal model, 3 epochs only)
- Model saved to `models/mnist.h5`
- All three endpoints verified: `/train`, `/status`, `/model`
