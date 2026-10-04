"""Optional integration check against the configured real local LLM.

Run from backend: .venv/bin/python scripts/smoke_local_model.py --mode ml
Uses an isolated temporary database and prints actual tool progress and metrics.
"""
import argparse
import asyncio
import tempfile
from pathlib import Path

from app.agent import run_analysis
from app.config import Settings
from app.datasets import register_dataset
from app.provider import OpenAICompatibleProvider
from app.storage import CHAT_DATASET_ID, Storage


async def main(mode):
    with tempfile.TemporaryDirectory(prefix="lumen-live-check-") as directory:
        settings = Settings(data_dir=Path(directory), max_tokens=4096, max_tool_calls=8, max_analysis_iterations=6)
        storage = Storage(settings.sqlite_path)
        storage.initialize()
        dataset_id = CHAT_DATASET_ID
        if mode != "chat":
            source = Path(__file__).resolve().parents[2] / "examples" / "ml_customers.csv"
            with source.open("rb") as file:
                dataset = register_dataset(filename=source.name, source=file, storage=storage, settings=settings)
            dataset_id = dataset.id
        if mode == "ml":
            request = ("Train a classification model to predict churned. Use only logistic regression and a baseline. "
                       "Exclude order_value. Prepare the dataset, check it, train, and report the actual holdout results. "
                       "Do not run predictions yet.")
        elif mode == "chat":
            request = "Research the latest Apache ECharts release using the web, cite the official source, and answer in three sentences."
        else:
            request = "What is the total monthly_spend? Use SQL and give a short answer."
        session = storage.create_session(dataset_id, "Local integration check", mode)
        storage.add_message(session.id, "user", request)
        query_seen, trained, searched, complete = False, False, False, False
        async for event in run_analysis(session_id=session.id, storage=storage, provider=OpenAICompatibleProvider(settings), settings=settings):
            if event.type in {"tool_started", "tool_finished", "ml_progress", "error"}:
                print(event.type, event.data.get("label", event.data.get("message", "")), event.data.get("error", ""), flush=True)
            if event.type == "query_result":
                query_seen = True
            if event.type == "tool_started" and event.data.get("tool") == "web_search":
                searched = True
            if event.type == "ml_result" and event.data.get("metrics"):
                trained = True
                print("Measured metrics:", event.data["metrics"], flush=True)
            if event.type == "analysis_complete":
                complete = True
                print("Answer:", event.data["message"], flush=True)
        expected = trained if mode == "ml" else searched if mode == "chat" else query_seen
        assert complete and expected, "The model did not complete the requested tool workflow"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sql", "ml", "chat"], default="ml")
    asyncio.run(main(parser.parse_args().mode))
