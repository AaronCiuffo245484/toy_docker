# src/ml_toy_repo/main.py
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from ml_toy_repo.state import JobStatus, job_state
from ml_toy_repo.trainer import MODEL_PATH, run_training

app = FastAPI(title="ML Toy Repo Training API")


@app.post("/train")
def start_training():
    if job_state.status == JobStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Training already in progress")
    job_state.reset()
    thread = threading.Thread(target=run_training, args=(job_state,), daemon=True)
    thread.start()
    return {"message": "Training started"}


@app.get("/status")
def get_status():
    return {
        "status": job_state.status.value,
        "error": job_state.error,
        "metrics": {
            "device": job_state.metrics.device,
            "epochs": job_state.metrics.epochs,
            "duration_seconds": job_state.metrics.duration_seconds,
            "final_accuracy": job_state.metrics.final_accuracy,
        },
    }


@app.get("/model")
def download_model():
    if not Path(MODEL_PATH).exists():
        raise HTTPException(status_code=404, detail="No model file found. Run /train first.")
    return FileResponse(
        path=MODEL_PATH,
        media_type="application/octet-stream",
        filename="mnist.h5",
    )