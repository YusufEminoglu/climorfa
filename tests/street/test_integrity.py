"""Unit tests for synthetic integrity checking tools."""

import tempfile
from pathlib import Path
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import yaml

from scripts.street.integrity.synthetic import assert_real, watermark, SYNTHETIC_TAG, WATERMARK_TEXT
from scripts.street.integrity.check_no_synthetic import check_file, check_manifest


def test_assert_real_filename_check():
    # Should raise on SYNTHETIC in path
    with pytest.raises(ValueError, match="synthetic/analogue marker"):
        assert_real("data/street/SYNTHETIC_kestrel.csv")

    with pytest.raises(ValueError, match="synthetic/analogue marker"):
        assert_real("data/street/met_ANALOGUE_nov.txt")

    # Clean path should not raise
    assert_real("data/street/field_kestrel_real.csv")


def test_assert_real_dataframe_check():
    # Synthetic dataframe
    df_synth = pd.DataFrame({"time": [1, 2], "is_synthetic": [True, True]})
    with pytest.raises(ValueError, match="is_synthetic=True"):
        assert_real(df_synth)

    # Real dataframe
    df_real = pd.DataFrame({"time": [1, 2], "is_synthetic": [False, False]})
    assert_real(df_real)


def test_watermark_applied():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    watermark(fig)

    # Check that text was added to figure or axes
    fig_texts = [t.get_text() for t in fig.texts]
    assert WATERMARK_TEXT in fig_texts

    ax_texts = [t.get_text() for t in ax.texts]
    assert WATERMARK_TEXT in ax_texts
    plt.close(fig)


def test_check_no_synthetic_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create a synthetic CSV
        synth_csv = tmppath / "SYNTHETIC_test.csv"
        pd.DataFrame({"x": [1], "is_synthetic": [True]}).to_csv(synth_csv, index=False)

        # Create a real CSV
        real_csv = tmppath / "real_obs.csv"
        pd.DataFrame({"x": [1], "is_synthetic": [False]}).to_csv(real_csv, index=False)

        # Create a CSV with clean name but is_synthetic=True
        sneaky_csv = tmppath / "station_data.csv"
        pd.DataFrame({"x": [1], "is_synthetic": [True]}).to_csv(sneaky_csv, index=False)

        # 1. Check clean manifest
        clean_manifest = tmppath / "manifest_clean.yaml"
        with open(clean_manifest, "w") as f:
            yaml.dump({"figures": {"fig1": [str(real_csv)]}}, f)

        is_clean, violations = check_manifest(str(clean_manifest))
        assert is_clean is True
        assert len(violations) == 0

        # 2. Check manifest with SYNTHETIC filename
        dirty_manifest = tmppath / "manifest_dirty.yaml"
        with open(dirty_manifest, "w") as f:
            yaml.dump({"figures": {"fig1": [str(synth_csv)]}}, f)

        is_clean, violations = check_manifest(str(dirty_manifest))
        assert is_clean is False
        assert any("SYNTHETIC" in v for v in violations)

        # 3. Check manifest with sneaky synthetic content
        sneaky_manifest = tmppath / "manifest_sneaky.yaml"
        with open(sneaky_manifest, "w") as f:
            yaml.dump({"tables": [str(sneaky_csv)]}, f)

        is_clean, violations = check_manifest(str(sneaky_manifest))
        assert is_clean is False
        assert any("is_synthetic=True" in v for v in violations)
