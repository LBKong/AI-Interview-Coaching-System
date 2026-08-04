"""Cold-process regression coverage for the FAISS/PyTorch load order."""

import subprocess
import sys
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_rag_primes_torch_before_faiss_in_cold_process():
    """Removing the torch-first guard must fail before native code can segfault."""
    code = textwrap.dedent(
        """
        import sys

        from server import rag

        assert "torch" in sys.modules, "server.rag must load torch first"
        assert "faiss" not in sys.modules, "FAISS must remain lazy"

        import faiss
        import torch

        matrix = torch.randn(256, 256)
        product = matrix @ matrix
        assert product.shape == (256, 256)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, (
        f"cold subprocess exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
