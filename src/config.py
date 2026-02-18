from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProjectConfig:
    """Central project configuration used by notebooks and pipeline modules."""

    root_dir: Path
    target_col: str = "readmitted"
    test_size: float = 0.20
    random_state: int = 42
    stratify_split: bool = True
    multiclass_map: dict[str, int] = field(
        default_factory=lambda: {"NO": 0, ">30": 1, "<30": 2}
    )
    binary_map: dict[str, int] = field(
        default_factory=lambda: {"NO": 0, ">30": 1, "<30": 1}
    )

    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data"

    @property
    def raw_data_path(self) -> Path:
        return self.data_dir / "raw" / "diabetic_data.csv"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def interim_train_path(self) -> Path:
        return self.interim_dir / "train.csv"

    @property
    def interim_test_path(self) -> Path:
        return self.interim_dir / "test.csv"

    @property
    def final_dir(self) -> Path:
        return self.data_dir / "final"

    @property
    def train_encoded_path(self) -> Path:
        return self.final_dir / "train_encoded.csv"

    @property
    def test_encoded_path(self) -> Path:
        return self.final_dir / "test_encoded.csv"

    @property
    def y_train_path(self) -> Path:
        return self.final_dir / "y_train.csv"

    @property
    def y_test_path(self) -> Path:
        return self.final_dir / "y_test.csv"

    @property
    def y_train_binary_path(self) -> Path:
        return self.final_dir / "y_train_binary.csv"

    @property
    def y_test_binary_path(self) -> Path:
        return self.final_dir / "y_test_binary.csv"

    @property
    def encoder_path(self) -> Path:
        return self.final_dir / "encoder.joblib"

    @property
    def model_path(self) -> Path:
        return self.final_dir / "best_model.joblib"

    @property
    def output_dir(self) -> Path:
        return self.root_dir / "output"

    @property
    def eda_dir(self) -> Path:
        return self.output_dir / "eda"

    @property
    def eda_figures_dir(self) -> Path:
        return self.eda_dir / "figures"

    @property
    def eda_tables_dir(self) -> Path:
        return self.eda_dir / "tables"

    def ensure_directories(self) -> None:
        """Create all required output directories."""
        required_dirs = [
            self.interim_dir,
            self.final_dir,
            self.eda_figures_dir,
            self.eda_tables_dir,
        ]
        for path in required_dirs:
            path.mkdir(parents=True, exist_ok=True)


def get_config(project_root: Path | None = None) -> ProjectConfig:
    """Build project config from the provided project root or auto-detected root."""
    root = project_root or Path(__file__).resolve().parents[1]
    config = ProjectConfig(root_dir=root)
    config.ensure_directories()
    return config
