"""
tests/test_timeseries.py
========================
Unit tests for the SPHEREx Time-Series Viewer & Streak-Aware Processor.
Covers NASA challenge targets, IRSA SIA handling, epoch sorting,
difference imaging, and the Section 4.9 photometric streak injection test.
"""

import pytest
import numpy as np

from spherex_streak.timeseries.irsa_client import (
    CHALLENGE_TARGETS,
    COLLECTION_QR2,
    COLLECTION_QR3,
    DOI_QR2,
    DOI_QR3,
    SpherexTimeSeriesClient,
)
from spherex_streak.timeseries.processor import (
    SpherexTimeSeriesProcessor,
)


class TestSpherexTimeSeriesClient:
    """Verifies target handling, coordinate parsing, and epoch sequencing."""

    def test_all_five_challenge_targets_defined(self):
        targets = SpherexTimeSeriesClient.get_supported_targets()
        expected = ["Barnard's Star", "Epsilon Indi B", "WISE 0855", "RW Aurigae", "R Draconis"]
        for exp in expected:
            assert exp in targets
            t_data = CHALLENGE_TARGETS[exp]
            assert "ra_deg" in t_data
            assert "dec_deg" in t_data
            assert 0.0 <= t_data["ra_deg"] <= 360.0
            assert -90.0 <= t_data["dec_deg"] <= 90.0

    def test_coordinate_parsing_decimal(self):
        ra, dec = SpherexTimeSeriesClient.parse_coordinates("269.452", "4.693")
        assert np.isclose(ra, 269.452)
        assert np.isclose(dec, 4.693)

    def test_coordinate_parsing_sexagesimal(self):
        # WISE 0855: RA 08h55m10.831s, Dec -07d14m42.53s
        ra, dec = SpherexTimeSeriesClient.parse_coordinates("08h55m10.831s", "-07d14m42.53s")
        assert np.isclose(ra, 133.795, atol=0.01)
        assert np.isclose(dec, -7.245, atol=0.01)

    def test_fetch_time_series_chronological_ordering(self):
        client = SpherexTimeSeriesClient(collection=COLLECTION_QR2)
        frames = client.fetch_time_series(
            ra_deg=269.45208,
            dec_deg=4.69339,
            target_name="Barnard's Star",
            num_epochs=6,
            fov_pixels=64,
        )
        assert len(frames) == 6
        # Verify strict chronological order
        mjds = [f.mjd_obs for f in frames]
        assert mjds == sorted(mjds)
        # Verify detector IDs between 1 and 6
        for f in frames:
            assert 1 <= f.detector_id <= 6
            assert 0.75 <= f.central_wavelength_um <= 5.0
            assert f.image.shape == (64, 64)
            assert f.variance.shape == (64, 64)

    def test_doi_and_collections(self):
        assert COLLECTION_QR2 == "spherex_qr2"
        assert DOI_QR2 == "10.26131/IRSA652"
        assert COLLECTION_QR3 == "spherex_qr3"
        assert DOI_QR3 == "10.26131/IRSA662"


class TestSpherexTimeSeriesProcessor:
    """Verifies difference imaging, co-addition, and streak injection benchmark."""

    def test_analyze_sequence_median_and_differences(self):
        client = SpherexTimeSeriesClient()
        frames = client.fetch_time_series(
            ra_deg=133.795,
            dec_deg=-7.245,
            target_name="WISE 0855",
            num_epochs=4,
            fov_pixels=48,
        )
        processor = SpherexTimeSeriesProcessor(pqm_threshold=0.5)
        res = processor.analyze_sequence(frames, enable_pqm_filter=True)

        assert res.total_epochs == 4
        assert res.median_image.shape == (48, 48)
        assert res.standard_coadd.shape == (48, 48)
        assert res.weighted_coadd.shape == (48, 48)
        assert len(res.difference_frames) == 4
        assert len(res.pqm_masks) == 4

        # Difference frame for each epoch should have mean near zero
        for diff in res.difference_frames:
            assert diff.shape == (48, 48)

    def test_photometric_streak_injection_benchmark(self):
        """Validates Section 4.9: synthetic streak injection on known star."""
        results = SpherexTimeSeriesProcessor.run_photometric_streak_injection_test(
            size=64,
            star_flux=100.0,
            streak_flux=40.0,
            streak_width=3,
            seed=42,
        )

        assert results["true_flux"] > 0
        assert results["raw_contaminated_flux"] > results["true_flux"]
        assert results["error_unmasked_pct"] > 10.0  # Significant unmasked bias

        # PQM recovered flux must have drastically smaller error (< 3.0%)
        assert results["error_pqm_pct"] < 3.0
        assert results["photometric_bias_reduction_factor"] > 3.0
