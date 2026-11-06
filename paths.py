"""Download links and local paths.

Fill in the four URLs before publishing the repository. The scripts read the
local paths below. They do not download the files.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# Paste the public download URLs here.
TRAIN_URL = ""
VAL_URL = ""
TEST_URL = ""
WEIGHT_URL = ""

# Local layout after the archives are unpacked.
SAM3_ROOT = ROOT / "third_party" / "sam3"
TRAIN_DIR = ROOT / "data" / "train"
VAL_DIR = ROOT / "data" / "val"
TEST_DIR = ROOT / "data" / "test"
SAM3_CHECKPOINT = ROOT / "weights" / "sam3.pt"
RGB_LORA = ROOT / "weights" / "rgb_lora.pt"
THERMAL_LORA = ROOT / "weights" / "thermal_lora.pt"
CHECKPOINT = ROOT / "weights" / "best.pth"


def require_file(path: Path, label: str, url: str, url_name: str) -> None:
    path = Path(path)
    if path.is_file():
        return
    if url.strip():
        detail = f"Download it using {url_name} in paths.py and place it at {path}."
    else:
        detail = f"Fill in {url_name} in paths.py, download the file, and place it at {path}."
    raise FileNotFoundError(f"Missing {label}. {detail}")


def require_dir(path: Path, label: str, url: str, url_name: str) -> None:
    path = Path(path)
    if path.is_dir():
        return
    if url.strip():
        detail = f"Download it using {url_name} in paths.py and unpack it at {path}."
    else:
        detail = f"Fill in {url_name} in paths.py, download the archive, and unpack it at {path}."
    raise FileNotFoundError(f"Missing {label}. {detail}")


def sam3_import_root() -> Path:
    root = Path(SAM3_ROOT)
    if (root / "model_builder.py").is_file():
        return root.parent
    raise FileNotFoundError(
        f"SAM3 package not found at {root}. Restore third_party/sam3 before running."
    )


def bpe_path() -> Path:
    path = Path(SAM3_ROOT) / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing SAM3 tokenizer vocabulary: {path}")
    return path
