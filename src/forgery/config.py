"""Locations of data, pretrained weights and outputs, configurable through environment variables."""
import os
from dataclasses import dataclass
from pathlib import Path

SEED = 42


@dataclass(frozen=True)
class Paths:
    """Dataset, pretrained-weight and output locations."""
    data_dir: Path
    output_dir: Path
    dinov2_dir: Path

    @classmethod
    def from_env(cls):
        """Build paths from FORGERY_DATA_DIR, FORGERY_OUTPUT_DIR and FORGERY_DINOV2_DIR."""
        return cls(
            data_dir=Path(os.environ.get("FORGERY_DATA_DIR", "data")),
            output_dir=Path(os.environ.get("FORGERY_OUTPUT_DIR", "outputs")),
            dinov2_dir=Path(os.environ.get("FORGERY_DINOV2_DIR", "weights/dinov2-base")),
        )

    @property
    def authentic(self):
        """Folder of authentic training images."""
        return self.data_dir / "train_images" / "authentic"

    @property
    def forged(self):
        """Folder of forged training images."""
        return self.data_dir / "train_images" / "forged"

    @property
    def masks(self):
        """Folder of training masks."""
        return self.data_dir / "train_masks"

    @property
    def supplemental(self):
        """Folder of supplemental images."""
        return self.data_dir / "supplemental_images"

    @property
    def supplemental_masks(self):
        """Folder of supplemental masks."""
        return self.data_dir / "supplemental_masks"
