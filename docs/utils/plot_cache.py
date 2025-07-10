"""Caching utilities for documentation plot generation."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any


class PlotCache:
    """A caching mechanism for plot generation that checks file hashes."""

    def __init__(self, cache_dir: Path):
        """Initialize the plot cache.

        Args:
            cache_dir: Directory to store cache metadata
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "plot_cache.pkl"
        self.cache_data = self._load_cache()

    def _load_cache(self) -> dict:
        """Load cache data from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)
            except (pickle.PickleError, EOFError):
                return {}
        return {}

    def _save_cache(self):
        """Save cache data to disk."""
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache_data, f)

    def _get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash of a file."""
        if not file_path.exists():
            return ""

        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def should_regenerate(self, source_file: Path, output_files: list[Path]) -> bool:
        """Check if plots should be regenerated.

        Args:
            source_file: The source file that generates the plots
            output_files: List of output image files

        Returns:
            True if plots should be regenerated, False otherwise
        """
        # Check if any output files are missing
        if not all(output_file.exists() for output_file in output_files):
            return True

        # Get current hash of source file
        current_hash = self._get_file_hash(source_file)

        # Check if hash has changed
        cache_key = str(source_file)
        if cache_key not in self.cache_data:
            return True

        cached_hash = self.cache_data[cache_key].get("hash", "")
        return current_hash != cached_hash

    def mark_generated(self, source_file: Path, output_files: list[Path]):
        """Mark plots as generated and update cache.

        Args:
            source_file: The source file that generates the plots
            output_files: List of output image files that were generated
        """
        current_hash = self._get_file_hash(source_file)
        cache_key = str(source_file)

        self.cache_data[cache_key] = {"hash": current_hash, "output_files": [str(f) for f in output_files]}

        self._save_cache()

    def cached_generation(self, source_file: Path, output_files: list[Path], generate_func: Callable[[], Any]) -> bool:
        """Execute generation function only if cache is invalid.

        Args:
            source_file: The source file that generates the plots
            output_files: List of output image files
            generate_func: Function to call to generate the plots

        Returns:
            True if plots were generated, False if cache was used
        """
        if self.should_regenerate(source_file, output_files):
            print(f"Regenerating plots for {source_file.name}")
            generate_func()
            self.mark_generated(source_file, output_files)
            return True
        else:
            print(f"Using cached plots for {source_file.name}")
            return False
