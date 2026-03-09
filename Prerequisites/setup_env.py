import subprocess
import os
import venv
import sys
from pathlib import Path

ENV_NAME = "training_env"
PYTHON_VERSION = "3.12"  # Force Python 3.12

PACKAGES = [
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "datasets",
    "numpy",
    "pandas",
    "pyarrow",
    "fsspec",
    "onnx",
    "onnxruntime",
    "openai",
    "python-dotenv",
    "requests",
    "langchain",
    "langchain-core",
    "langchain-openai",
    "langchain-community",
    "langchain-experimental",
    "langchain-text-splitters",
    "langchain-chroma",
    "sentence-transformers",
    "faiss-cpu",
    "chromadb",
    "pypdf",
    "docx2txt",
    "python-docx",
    "docling",
    "docling-core",
    "rank-bm25",
    "tiktoken",
    "nltk",
    "easyocr",
    "matplotlib",
    "psutil",
    "urllib3",
    "botocore",
    "ipykernel",
    "notebook",
    "pydantic"
]


def run(cmd):
    print("..", " ".join(cmd))
    subprocess.check_call(cmd)


def python_path():
    if os.name == "nt":
        return Path(ENV_NAME) / "Scripts" / "python"
    return Path(ENV_NAME) / "bin" / "python"


def create_env():
    if not Path(ENV_NAME).exists():
        print(f"Creating virtual environment with Python {PYTHON_VERSION}...")

        # Use 'py' launcher to find Python 3.12
        try:
            # Try to get Python 3.12 executable using py launcher
            python_exe = subprocess.check_output(
                ["py", f"-{PYTHON_VERSION}", "-c",
                    "import sys; print(sys.executable)"],
                text=True
            ).strip()
            print(f"Found Python 3.12 at: {python_exe}")
            print(python_exe)
            # Create venv with the specific Python version
            venv.create(ENV_NAME, executable=python_exe, with_pip=True)
            # venv.create(ENV_NAME, with_pip=True)

        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"ERROR: Python {PYTHON_VERSION} not found!")
            print("Install Python 3.12 from https://www.python.org/ or Microsoft Store")
            print("Or use: winget install Python.Python.3.12")
            sys.exit(1)
    else:
        print("Virtual environment already exists")


def install_packages():
    py = str(python_path())

    # Correct way to upgrade pip on Windows
    run([py, "-m", "pip", "install", "--upgrade", "pip"])

    # Install all required packages
    run([py, "-m", "pip", "install", *PACKAGES])


def setup_nltk():
    py = str(python_path())
    run([py, "-c", "import nltk; nltk.download('punkt')"])


if __name__ == "__main__":
    print("Training Environment Setup Started\n")
    print(f"Target Python Version: {PYTHON_VERSION}")
    print(f"Current Python Version: {sys.version}")
    print("-" * 60)

    create_env()
    install_packages()
    setup_nltk()

    py_path = python_path()
    print(f"\n✅ Setup complete!")
    print(f"\n📍 Virtual environment location: {Path(ENV_NAME).absolute()}")
    print(f"\n🔧 NEXT STEP:")
    print("VS Code → Ctrl+Shift+P → Python: Select Interpreter")
    print(f"Choose: {ENV_NAME}")
