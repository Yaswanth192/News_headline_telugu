"""
src/inference/manager.py
========================
Phase 6 — Memory-Safe Model Manager & Inference Engine for Telugu News Headline Studio.

Guarantees:
- Strict local GPU resource management for 4 GB VRAM RTX 3050.
- Lazy-loading of checkpoints; single active transformer in CUDA memory.
- Explicit cleanup (del model, gc.collect(), torch.cuda.empty_cache()).
- Supports mT5-small (all checkpoints), IndicBART, and mBERT / TF-IDF relevance classifiers.
- Generates predictions with latency and token stats.
"""

import gc
import logging
import os
import pathlib
import sys
import time
from typing import Dict, Any, Optional, Tuple

import torch

# Fix Windows cp1252 for Telugu characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Registry of available models and checkpoints
MODEL_REGISTRY = {
    "mt5_small_final": {
        "name": "mT5-small (Phase 1/3 Final)",
        "type": "mt5",
        "path": PROJECT_ROOT / "checkpoints" / "mt5_small" / "final",
        "description": "mT5-small baseline fine-tuned on full Telugu news dataset (Phase 1 & Phase 3)",
        "architecture": "mT5ForConditionalGeneration",
    },
    "mt5_small_step50": {
        "name": "mT5-small (Step 50 Intermediate)",
        "type": "mt5",
        "path": PROJECT_ROOT / "checkpoints" / "mt5_small" / "step_50",
        "description": "Intermediate checkpoint of mT5-small baseline after 50 steps",
        "architecture": "mT5ForConditionalGeneration",
    },
    "indicbart_final": {
        "name": "IndicBART (Phase 3 Final)",
        "type": "indicbart",
        "path": PROJECT_ROOT / "checkpoints" / "indicbart" / "final",
        "description": "IndicBART compact seq2seq fine-tuned with Telugu language tokens (<2te>)",
        "architecture": "MBartForConditionalGeneration",
    },
    "low_resource_5pct": {
        "name": "mT5-small (5% Data - 75 Steps)",
        "type": "mt5",
        "path": PROJECT_ROOT / "checkpoints" / "low_resource" / "5pct" / "final",
        "description": "Low-resource experiment: trained on 5% deterministic subset (126 samples)",
        "architecture": "mT5ForConditionalGeneration",
    },
    "low_resource_25pct": {
        "name": "mT5-small (25% Data - 75 Steps)",
        "type": "mt5",
        "path": PROJECT_ROOT / "checkpoints" / "low_resource" / "25pct" / "final",
        "description": "Low-resource experiment: trained on 25% deterministic subset (634 samples)",
        "architecture": "mT5ForConditionalGeneration",
    },
    "low_resource_100pct": {
        "name": "mT5-small (100% Data - 75 Steps)",
        "type": "mt5",
        "path": PROJECT_ROOT / "checkpoints" / "low_resource" / "100pct" / "final",
        "description": "Low-resource controlled baseline: trained on 100% dataset for 75 steps (2,538 samples)",
        "architecture": "mT5ForConditionalGeneration",
    },
}

CLASSIFIER_REGISTRY = {
    "bert": {
        "name": "mBERT Headline Relevance Classifier (Phase 2)",
        "type": "bert",
        "path": PROJECT_ROOT / "checkpoints" / "classifier_bert" / "final",
        "description": "Multilingual BERT fine-tuned on TeClass dataset (HREL / MREL / LREL)",
    },
    "tfidf": {
        "name": "TF-IDF + Logistic Regression (Phase 2 Baseline)",
        "type": "tfidf",
        "path": PROJECT_ROOT / "checkpoints" / "classifier_tfidf" / "tfidf_logreg.pkl",
        "description": "Linear classifier baseline using character n-grams and Telugu TF-IDF",
    },
}


class ModelManager:
    """Manages sequential loading, unloading, and execution of generation and classification models."""

    def __init__(self, force_cpu: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
        self.active_gen_key: Optional[str] = None
        self.active_gen_model = None
        self.active_gen_tokenizer = None

    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Return real-time VRAM allocation, reservation, and physical capacity."""
        if not torch.cuda.is_available() or self.device.type == "cpu":
            return {
                "available": False,
                "device_name": "CPU",
                "allocated_mib": 0.0,
                "reserved_mib": 0.0,
                "total_mib": 0.0,
            }
        
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        return {
            "available": True,
            "device_name": torch.cuda.get_device_name(0),
            "allocated_mib": round(allocated, 2),
            "reserved_mib": round(reserved, 2),
            "total_mib": round(total, 2),
            "free_approx_mib": round(total - reserved, 2),
        }

    def unload_active_generator(self):
        """Unload active generation model and reclaim GPU memory."""
        if self.active_gen_model is not None:
            logger.info("Unloading generation model: %s", self.active_gen_key)
            del self.active_gen_model
            del self.active_gen_tokenizer
            self.active_gen_model = None
            self.active_gen_tokenizer = None
            self.active_gen_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            logger.info("GPU VRAM cleared.")

    def load_generator(self, model_key: str):
        """Load specified generation model into GPU/CPU memory safely."""
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")

        if self.active_gen_key == model_key and self.active_gen_model is not None:
            return self.active_gen_model, self.active_gen_tokenizer

        # Always unload previous model first to respect 4 GB VRAM limit
        self.unload_active_generator()

        meta = MODEL_REGISTRY[model_key]
        ckpt_path = meta["path"]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

        logger.info("Loading %s from %s onto %s...", meta["name"], ckpt_path, self.device)
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path), use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt_path))
        model.to(self.device)
        model.eval()

        self.active_gen_key = model_key
        self.active_gen_model = model
        self.active_gen_tokenizer = tokenizer
        return self.active_gen_model, self.active_gen_tokenizer

    def generate(
        self,
        model_key: str,
        article_text: str,
        max_length: int = 64,
        num_beams: int = 4,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> Dict[str, Any]:
        """Generate a headline for an input Telugu article text."""
        article_text = (article_text or "").strip()
        if not article_text:
            raise ValueError("Input article text cannot be empty.")

        model, tokenizer = self.load_generator(model_key)
        meta = MODEL_REGISTRY[model_key]
        model_type = meta["type"]

        start_time = time.perf_counter()

        import re
        if model_type == "mt5":
            source = "summarize: " + article_text
            inputs = tokenizer(
                source,
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            decoder_start = getattr(tokenizer, "pad_token_id", 0) or 0

            gen_kwargs = {
                "max_new_tokens": max_length,
                "num_beams": num_beams,
                "decoder_start_token_id": decoder_start,
                "early_stopping": True,
                "no_repeat_ngram_size": 3,
            }
            if do_sample:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = max(0.1, temperature)
                gen_kwargs.pop("num_beams", None)

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            # Clean sentinel tokens if any remain
            clean_headline = re.sub(r"<extra_id_\d+>", "", decoded).strip()
            clean_headline = re.sub(r"\s+", " ", clean_headline)

        elif model_type == "indicbart":
            source = article_text + " <2te>"
            inputs = tokenizer(
                source,
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            inputs.pop("token_type_ids", None)

            gen_kwargs = {
                "max_new_tokens": max_length,
                "num_beams": num_beams,
                "early_stopping": True,
                "no_repeat_ngram_size": 3,
            }
            if hasattr(tokenizer, "lang_code_to_id") and "<2te>" in tokenizer.lang_code_to_id:
                gen_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id["<2te>"]

            if do_sample:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = max(0.1, temperature)
                gen_kwargs.pop("num_beams", None)

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            clean_headline = re.sub(r"\[CLS\]|\[SEP\]|<2te>", "", decoded).strip()
            clean_headline = re.sub(r"\s+", " ", clean_headline)
        else:
            raise NotImplementedError(f"Unsupported model type: {model_type}")

        latency = time.perf_counter() - start_time
        token_count = len(clean_headline.split())

        return {
            "model_key": model_key,
            "model_name": meta["name"],
            "headline": clean_headline,
            "raw_output": decoded,
            "latency_seconds": round(latency, 3),
            "generated_words": token_count,
            "device": str(self.device),
        }

    def classify_relevance(self, article: str, headline: str, model_type: str = "bert") -> Dict[str, Any]:
        """Classify relevance of a headline with respect to the article using Phase 2 inference.
        
        To preserve VRAM, before loading BERT classifier, we unload any generator if running on CUDA.
        """
        if self.device.type == "cuda" and model_type == "bert":
            self.unload_active_generator()

        from src.classification.predict import predict_relevance
        result = predict_relevance(article=article, headline=headline, model_type=model_type)
        return result


# Singleton manager instance
_MANAGER: Optional[ModelManager] = None


def get_model_manager(force_cpu: bool = False) -> ModelManager:
    global _MANAGER
    if _MANAGER is None or (force_cpu and _MANAGER.device.type != "cpu"):
        _MANAGER = ModelManager(force_cpu=force_cpu)
    return _MANAGER
