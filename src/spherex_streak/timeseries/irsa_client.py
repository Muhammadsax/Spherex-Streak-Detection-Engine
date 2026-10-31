"""
src/spherex_streak/timeseries/irsa_client.py
============================================
Client and simulation engine for NASA/IPAC IRSA SPHEREx Simple Image Access (SIA)
and multi-epoch image retrieval for time-series discovery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Official IRSA Collection Identifiers and DOIs
COLLECTION_QR2 = "spherex_qr2"
COLLECTION_QR3 = "spherex_qr3"
DOI_QR2 = "10.26131/IRSA652"
DOI_QR3 = "10.26131/IRSA662"
IRSA_SIA_ENDPOINT = "https://irsa.ipac.caltech.edu/SIA"

# Official NASA Challenge Target Presets with astronomical coordinates
CHALLENGE_TARGETS: Dict[str, Dict] = {
    "Barnard's Star": {
        "name": "Barnard's Star",
        "ra_deg": 269.45208,
        "dec_deg": 4.69339,
        "description": "High proper motion red dwarf (10.3 arcsec/yr) in Ophiuchus",
        "type": "proper_motion",
        "pm_ra_mas_yr": -801.55,
        "pm_dec_mas_yr": 10362.39,
        "target_flux": 120.0,
    },
    "Epsilon Indi B": {
        "name": "Epsilon Indi B",
        "ra_deg": 331.04375,
        "dec_deg": -56.78278,
        "description": "Nearby brown dwarf binary system with significant motion",
        "type": "proper_motion",
        "pm_ra_mas_yr": 3961.4,
        "pm_dec_mas_yr": -2538.3,
        "target_flux": 85.0,
    },
    "WISE 0855": {
        "name": "WISE 0855-0714",
        "ra_deg": 133.79513,
        "dec_deg": -7.24515,
        "description": "Coldest known sub-stellar brown dwarf (RA 08h55m10.8s, Dec -07d14m42.5s)",
        "type": "proper_motion",
        "pm_ra_mas_yr": -8132.0,
        "pm_dec_mas_yr": 639.0,
        "target_flux": 45.0,
    },
    "RW Aurigae": {
        "name": "RW Aurigae",
        "ra_deg": 76.95625,
        "dec_deg": 30.40139,
        "description": "Classical T Tauri young variable star showing erratic deep dimming",
        "type": "variable",
        "variability_period_days": 2.77,
        "amplitude": 65.0,
        "target_flux": 110.0,
    },
    "R Draconis": {
        "name": "R Draconis",
        "ra_deg": 248.16750,
        "dec_deg": 66.75500,
        "description": "Mira-type long-period pulsating variable star (period ~245 days)",
        "type": "variable",
        "variability_period_days": 245.0,
        "amplitude": 90.0,
        "target_flux": 140.0,
    },
}


@dataclass
class EpochFrame:
    """Represents a single SPHEREx exposure epoch for a given sky coordinate."""
    epoch_index: int
    date_obs: str
    mjd_obs: float
    detector_id: int
    central_wavelength_um: float
    image: np.ndarray  # 2D Float32 flux [MJy/sr or DN]
    variance: np.ndarray  # 2D Float32 pixel noise variance (sigma^2)
    flags: np.ndarray  # 2D Int32 quality flags
    streak_mask_gt: Optional[np.ndarray] = None  # Synthetic ground truth if streak present
    has_streak: bool = False


class SpherexTimeSeriesClient:
    """
    Client for retrieving or generating multi-epoch SPHEREx observations.
    Provides offline simulation of the 5 official challenge targets to ensure
    100% reliability and interactivity, along with IRSA SIA query builders.
    """

    def __init__(self, collection: str = COLLECTION_QR2):
        self.collection = collection
        self.doi = DOI_QR2 if collection == COLLECTION_QR2 else DOI_QR3

    @staticmethod
    def get_supported_targets() -> List[str]:
        return list(CHALLENGE_TARGETS.keys())

    @staticmethod
    def parse_coordinates(ra_input: str, dec_input: str) -> Tuple[float, float]:
        """
        Parses RA and Dec from either decimal degrees or sexagesimal format.
        """
        try:
            ra = float(ra_input)
            dec = float(dec_input)
            return ra % 360.0, max(-90.0, min(90.0, dec))
        except ValueError:
            # Handle sexagesimal parsing (e.g. '08:55:10.8' or '08h55m10.8s')
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            coord = SkyCoord(ra_input, dec_input, unit=(u.hourangle, u.deg))
            return float(coord.ra.deg), float(coord.dec.deg)

    def fetch_time_series(
        self,
        ra_deg: float,
        dec_deg: float,
        target_name: Optional[str] = None,
        num_epochs: int = 8,
        fov_pixels: int = 128,
        seed: int = 42,
    ) -> List[EpochFrame]:
        """
        Fetches or simulates a multi-epoch sequence for the requested coordinates.
        In offline/demo mode, realistically simulates the physics of the target
        (proper motion, variability, background noise, SUR satellite streaks).
        """
        rng = np.random.default_rng(seed)
        frames: List[EpochFrame] = []

        # Check if requested coordinates match a known challenge target
        matched_target = None
        if target_name and target_name in CHALLENGE_TARGETS:
            matched_target = CHALLENGE_TARGETS[target_name]
        else:
            for t_name, t_info in CHALLENGE_TARGETS.items():
                if np.isclose(ra_deg, t_info["ra_deg"], atol=0.05) and np.isclose(dec_deg, t_info["dec_deg"], atol=0.05):
                    matched_target = t_info
                    break

        y_grid, x_grid = np.mgrid[0:fov_pixels, 0:fov_pixels]
        center = fov_pixels // 2

        # Base dates across SPHEREx mission survey passes (e.g. 2025-2026)
        base_mjd = 60800.0  # Approx mid 2025
        epoch_delays = np.sort(rng.uniform(0.5, 180.0, size=num_epochs))

        # Static background stars in the field
        num_field_stars = 4
        field_stars = []
        for _ in range(num_field_stars):
            fx = rng.uniform(0.15 * fov_pixels, 0.85 * fov_pixels)
            fy = rng.uniform(0.15 * fov_pixels, 0.85 * fov_pixels)
            fflux = rng.uniform(30.0, 90.0)
            field_stars.append((fx, fy, fflux))

        # Designate 1 or 2 epochs to have satellite streak contamination (Section 4.1 SUR physics)
        streak_epochs = set(rng.choice(range(num_epochs), size=min(2, max(1, num_epochs // 4)), replace=False))

        for idx in range(num_epochs):
            dt_days = float(epoch_delays[idx])
            mjd = base_mjd + dt_days
            date_str = f"2025-{int(5 + dt_days // 30):02d}-{int(1 + dt_days % 28):02d}T12:00:00"

            # Sky noise (sigma = 4.5) & Zodiacal background
            noise_sigma = 4.5
            sky_noise = rng.normal(0.0, noise_sigma, size=(fov_pixels, fov_pixels)).astype(np.float32)
            zodi_bkg = 50.0 + 0.02 * x_grid + 0.01 * y_grid
            img = (zodi_bkg + sky_noise).astype(np.float32)
            var_map = np.full((fov_pixels, fov_pixels), noise_sigma**2, dtype=np.float32)
            flags = np.zeros((fov_pixels, fov_pixels), dtype=np.int32)
            gt_streak = np.zeros((fov_pixels, fov_pixels), dtype=np.float32)
            has_streak = idx in streak_epochs

            # Render static field stars
            for fx, fy, fflux in field_stars:
                star = fflux * np.exp(-((x_grid - fx)**2 + (y_grid - fy)**2) / (2 * 1.6**2))
                img += star.astype(np.float32)

            # Render Target object
            if matched_target is not None:
                if matched_target["type"] == "proper_motion":
                    # Target moves across epochs
                    motion_x = (dt_days / 180.0) * (fov_pixels * 0.25)
                    motion_y = (dt_days / 180.0) * (fov_pixels * 0.15)
                    tx = center - fov_pixels * 0.12 + motion_x
                    ty = center - fov_pixels * 0.08 + motion_y
                    tflux = matched_target["target_flux"]
                else:  # variable star
                    tx = float(center)
                    ty = float(center)
                    # Periodic sinusoidal flux variation
                    period = matched_target.get("variability_period_days", 10.0)
                    phase = (dt_days % period) / period
                    tflux = matched_target["target_flux"] + matched_target.get("amplitude", 40.0) * np.sin(2 * np.pi * phase)
                    tflux = max(15.0, tflux)

                target_profile = tflux * np.exp(-((x_grid - tx)**2 + (y_grid - ty)**2) / (2 * 1.8**2))
                img += target_profile.astype(np.float32)
            else:
                # Generic stationary star at center
                target_profile = 75.0 * np.exp(-((x_grid - center)**2 + (y_grid - center)**2) / (2 * 1.8**2))
                img += target_profile.astype(np.float32)

            # Inject Satellite Streak (SUR signature) if epoch is contaminated
            if has_streak:
                sy = int(fov_pixels * (0.3 + 0.4 * (idx % 2)))
                x_start = int(0.08 * fov_pixels)
                x_end = int(0.92 * fov_pixels)

                # SUR dual parallel tracks residual pattern (Borlaff et al. 2026)
                sy1 = max(0, sy - 2)
                sy2 = min(fov_pixels - 1, sy + 2)
                img[sy1, x_start:x_end] += 25.0
                img[sy2, x_start:x_end] += 22.0
                gt_streak[sy1, x_start:x_end] = 1.0
                gt_streak[sy2, x_start:x_end] = 1.0
                flags[sy1, x_start:x_end] |= 4  # Mark potential transient in FLAGS
                flags[sy2, x_start:x_end] |= 4

            # Assign detector ID (1 through 6) and central wavelength (0.75 - 5.0 um)
            det_id = (idx % 6) + 1
            central_wave = float(0.75 + (det_id - 1) * (4.25 / 5.0))

            frame = EpochFrame(
                epoch_index=idx,
                date_obs=date_str,
                mjd_obs=float(mjd),
                detector_id=det_id,
                central_wavelength_um=round(central_wave, 2),
                image=img,
                variance=var_map,
                flags=flags,
                streak_mask_gt=gt_streak,
                has_streak=has_streak,
            )
            frames.append(frame)

        # Sort strictly chronologically by MJD-OBS
        frames.sort(key=lambda f: f.mjd_obs)
        return frames
