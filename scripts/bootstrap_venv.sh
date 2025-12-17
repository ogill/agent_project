#!/usr/bin/env bash
set -e

echo "🔧 Bootstrapping agent_project Python environment..."

# --- Config ---
PYTHON_VERSION="3.12.8"
VENV_DIR=".venv"

# --- Ensure pyenv is available ---
if ! command -v pyenv >/dev/null 2>&1; then
  echo "❌ pyenv not found. Install pyenv first."
  exit 1
fi

# --- Initialise pyenv (important for non-login shells) ---
eval "$(pyenv init -)"

# --- Install Python if missing ---
if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
  echo "📦 Installing Python ${PYTHON_VERSION} via pyenv..."
  pyenv install "${PYTHON_VERSION}"
else
  echo "✅ Python ${PYTHON_VERSION} already installed"
fi

# --- Set local Python version ---
pyenv local "${PYTHON_VERSION}"

# --- Rehash shims ---
pyenv rehash

# --- Remove old venv if present ---
if [ -d "${VENV_DIR}" ]; then
  echo "🧹 Removing existing ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

# --- Create venv ---
echo "🐍 Creating virtual environment..."
python -m venv "${VENV_DIR}"

# --- Activate venv ---
source "${VENV_DIR}/bin/activate"

# --- Upgrade core tooling ---
echo "⬆️  Upgrading pip / setuptools / wheel..."
python -m pip install -U pip setuptools wheel

# --- Install runtime deps ---
echo "📦 Installing runtime dependencies..."
pip install \
  chromadb \
  requests

# --- Install dev/test deps ---
echo "🧪 Installing dev dependencies..."
pip install pytest

echo ""
echo "✅ Environment ready!"
echo "👉 Activate with: source .venv/bin/activate"
echo "👉 Run tests with: pytest -q"
echo "👉 Run app with: python main.py"
