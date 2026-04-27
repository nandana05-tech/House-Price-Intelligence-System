"""
Singleton model loader — loads all ML artifacts once at startup.
Call `models.load()` before using any model attribute.
"""
import json
import pickle
from pathlib import Path

from catboost import CatBoostClassifier, CatBoostRegressor

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
METADATA_DIR = BASE_DIR / "metadata"


class ModelLoader:
    """Thread-safe singleton that holds all loaded ML artifacts."""

    _instance: "ModelLoader | None" = None
    _loaded: bool = False

    def __new__(cls) -> "ModelLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load every artifact from disk. Idempotent — safe to call multiple times."""
        if self._loaded:
            return

        print("[ModelLoader] Loading all artifacts...")

        # ── Regression ────────────────────────────────────────────────
        self.model_low = self._load_catboost(CatBoostRegressor, MODELS_DIR / "model_low.cbm")
        self.model_high = self._load_catboost(CatBoostRegressor, MODELS_DIR / "model_high.cbm")

        # ── Classification ────────────────────────────────────────────
        self.model_clf = self._load_catboost(CatBoostClassifier, MODELS_DIR / "model_clf.cbm")

        # ── Clustering pipeline ───────────────────────────────────────
        self.kmeans = self._load_pickle(MODELS_DIR / "kmeans_model.pkl")
        self.umap = self._load_pickle(MODELS_DIR / "umap_reducer.pkl")
        self.scaler = self._load_pickle(MODELS_DIR / "scaler.pkl")

        # ── Target encoder (shared by regression & clustering) ────────
        self.target_encoder = self._load_pickle(MODELS_DIR / "target_encoder.pkl")

        # ── Metadata ──────────────────────────────────────────────────
        self.meta_regresi: dict = self._load_json(METADATA_DIR / "metadata_regresi.json")
        self.meta_klasifikasi: dict = self._load_json(METADATA_DIR / "metadata_klasifikasi.json")
        self.meta_clustering: dict = self._load_json(METADATA_DIR / "metadata_clustering.json")

        self._loaded = True
        print("[ModelLoader] All artifacts loaded successfully.")

    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_pickle(self, path: Path) -> object:
        if not path.exists():
            raise FileNotFoundError(f"[ModelLoader] Artifact not found: {path}")
        try:
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
            print(f"[ModelLoader] Loaded: {path.name}")
            return obj
        except Exception as e:
            raise RuntimeError(f"[ModelLoader] Failed to load {path.name}: {e}") from e

    def _load_catboost(self, cls, path: Path) -> object:
        if not path.exists():
            raise FileNotFoundError(f"[ModelLoader] Model not found: {path}")
        try:
            model = cls()
            model.load_model(str(path))
            print(f"[ModelLoader] Loaded: {path.name}")
            return model
        except Exception as e:
            raise RuntimeError(f"[ModelLoader] Failed to load {path.name}: {e}") from e

    def _load_json(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"[ModelLoader] Metadata not found: {path}")
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            print(f"[ModelLoader] Loaded: {path.name}")
            return data
        except Exception as e:
            raise RuntimeError(f"[ModelLoader] Failed to load {path.name}: {e}") from e


# Module-level singleton — import this everywhere
models = ModelLoader()