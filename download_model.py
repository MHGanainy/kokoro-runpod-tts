#!/usr/bin/env python3
"""Download and prepare Kokoro v1.0 model."""

import json
import os
from pathlib import Path
from urllib.request import urlretrieve


def verify_files(model_path: str, config_path: str) -> bool:
    """Verify that model files exist and are valid.

    Args:
        model_path: Path to model file
        config_path: Path to config file

    Returns:
        True if files exist and are valid
    """
    try:
        # Check files exist
        if not os.path.exists(model_path):
            return False
        if not os.path.exists(config_path):
            return False

        # Verify config file is valid JSON
        with open(config_path) as f:
            config = json.load(f)

        # Check model file size (should be non-zero)
        if os.path.getsize(model_path) == 0:
            return False

        return True
    except Exception:
        return False


def download_model(output_dir: str) -> None:
    """Download model files from GitHub release.

    Args:
        output_dir: Directory to save model files
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Define file paths
        model_file = "kokoro-v1_0.pth"
        config_file = "config.json"
        model_path = os.path.join(output_dir, model_file)
        config_path = os.path.join(output_dir, config_file)

        # Check if files already exist and are valid
        if verify_files(model_path, config_path):
            print("Model files already exist and are valid")
            return

        print("Downloading Kokoro v1.0 model files")

        # GitHub release URLs
        base_url = "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.4"
        model_url = f"{base_url}/{model_file}"
        config_url = f"{base_url}/{config_file}"

        # Download files
        print("Downloading model file...")
        urlretrieve(model_url, model_path)

        print("Downloading config file...")
        urlretrieve(config_url, config_path)

        # Verify downloaded files
        if not verify_files(model_path, config_path):
            raise RuntimeError("Failed to verify downloaded files")

        print(f"✓ Model files prepared in {output_dir}")

    except Exception as e:
        print(f"Failed to download model: {e}")
        raise


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Download Kokoro v1.0 model")
    parser.add_argument(
        "--output", required=True, help="Output directory for model files"
    )

    args = parser.parse_args()
    download_model(args.output)


if __name__ == "__main__":
    main()