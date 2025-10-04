import argparse
import numpy as np
import pandas as pd
import lightkurve as lk
from scipy import interpolate
import warnings
import time
import os

# --- Suppress Warnings ---
warnings.filterwarnings('ignore', category=lk.LightkurveWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class KeplerPreprocessor:
    """Preprocessing pipeline for Kepler/TESS/K2 light curves (Shallue & Vanderburg 2018)."""

    def __init__(self, period, t0, duration, mission="Kepler", max_retries=3, retry_delay=5):
        self.period = period
        self.t0 = t0
        self.duration_days = duration / 24.0
        self.mission = mission
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def download_lightcurve(self, target_id):
        """Downloads and stitches all available light curves for a target."""
        for attempt in range(1, self.max_retries + 1):
            try:
                search = lk.search_lightcurve(target_id, mission=self.mission)
                if not search:
                    warnings.warn(f"No light curves found for {target_id} ({self.mission})")
                    return None
                lc_collection = search.download_all()
                if not lc_collection:
                    return None
                return lc_collection.stitch().remove_nans()
            except Exception as e:
                print(f"-> [Attempt {attempt}/{self.max_retries}] Download for {target_id} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        return None

    def flatten_lightcurve(self, lc):
        """Flattens the light curve using a spline fit with a robust fallback."""
        try:
            time, flux = lc.time.value, lc.flux.value
            phase = ((time - self.t0 + 0.5 * self.period) % self.period) - 0.5 * self.period
            in_transit = np.abs(phase) < self.duration_days
            best_bic, best_spline = np.inf, None
            spacings = [0.5, 0.75, 1.0, 1.5, 2.0]

            for spacing in spacings:
                n_knots = int(np.ceil((time[-1] - time[0]) / spacing))
                if n_knots < 4:
                    continue
                knots = np.linspace(time[0], time[-1], n_knots)
                mask = ~in_transit
                spline = None
                for _ in range(3):
                    try:
                        if np.sum(mask) <= 3:
                            break
                        spline = interpolate.LSQUnivariateSpline(time[mask], flux[mask], knots[1:-1], k=3)
                        residuals = flux[mask] - spline(time[mask])
                        outliers = np.abs(residuals) > 3 * np.std(residuals)
                        mask[np.where(mask)[0][outliers]] = False
                    except Exception:
                        spline = None
                        break
                if spline:
                    residuals = flux[mask] - spline(time[mask])
                    n, k = len(time[mask]), len(knots)
                    bic = n * np.log(np.sum(residuals**2) / n) + k * np.log(n)
                    if bic < best_bic:
                        best_bic, best_spline = bic, spline

            if best_spline is None:
                raise ValueError("Spline fitting failed")

            return flux / best_spline(time), time

        except Exception as e:
            warnings.warn(f"Primary spline fit failed ('{e}'). Using fallback flatten.")
            window_length = int(20 * self.duration_days / np.median(np.diff(lc.time.value)))
            window_length = max(101, window_length + (1 - window_length % 2))
            flattened_lc = lc.flatten(window_length=window_length)
            return flattened_lc.flux.value, flattened_lc.time.value

    def fold_and_bin(self, time, flux, n_bins, view_span_days):
        """Folds and bins the light curve."""
        phase = (((time - self.t0 + 0.5 * self.period) % self.period) / self.period)
        phase[phase > 0.5] -= 1.0
        phase_time = phase * self.period
        bin_centers = np.linspace(-view_span_days / 2, view_span_days / 2, n_bins)
        bin_width = np.abs(bin_centers[1] - bin_centers[0])
        binned_flux = np.zeros(n_bins)
        for i, center in enumerate(bin_centers):
            in_bin = np.abs(phase_time - center) < bin_width / 2
            binned_flux[i] = np.median(flux[in_bin]) if np.any(in_bin) else np.nan
        if np.any(np.isnan(binned_flux)):
            mask = ~np.isnan(binned_flux)
            if np.sum(mask) > 1:
                binned_flux = np.interp(bin_centers, bin_centers[mask], binned_flux[mask])
            else:
                binned_flux = np.nan_to_num(binned_flux, nan=1.0)
        return binned_flux

    def normalize(self, arr):
        """Shifts median to 0 and scales minimum to -1."""
        arr = arr - np.median(arr)
        min_val = np.min(arr)
        if min_val < -1e-9:
            arr /= -min_val
        return arr

    def preprocess(self, lc):
        """Full preprocessing pipeline for a single light curve."""
        if lc is None:
            raise ValueError("LightCurve object is None.")
        flux, time = self.flatten_lightcurve(lc)
        global_view = self.fold_and_bin(time, flux, n_bins=2001, view_span_days=self.period)
        local_view = self.fold_and_bin(time, flux, n_bins=201, view_span_days=8 * self.duration_days)
        return self.normalize(global_view), self.normalize(local_view)


def process_single_target(target_id, period, t0, duration, mission):
    """Wrapper to process a single target."""
    print(f"\nProcessing {target_id} ({mission})...")
    preprocessor = KeplerPreprocessor(period, t0, duration, mission)
    lc = preprocessor.download_lightcurve(target_id)
    return preprocessor.preprocess(lc)


def process_batch(csv_path):
    """
    Processes a batch of targets from a CSV file.
    Expected columns: target_id, mission, period, t0, duration
    """
    df = pd.read_csv(csv_path)
    required_cols = {"target_id", "mission", "period", "t0", "duration"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")

    global_views, local_views = [], []
    metadata_records = []

    for _, row in df.iterrows():
        try:
            global_view, local_view = process_single_target(
                row["target_id"], row["period"], row["t0"], row["duration"], row["mission"]
            )
            global_views.append(global_view)
            local_views.append(local_view)
            metadata_records.append(row.to_dict())
            print(f"✅ {row['target_id']} processed successfully.")
        except Exception as e:
            print(f"⚠️ Skipping {row['target_id']} due to error: {e}")

    # Save results
    os.makedirs("batch_output", exist_ok=True)
    pd.DataFrame(metadata_records).to_csv("batch_output/batch_metadata.csv", index=False)
    pd.DataFrame(global_views).to_csv("batch_output/batch_global_views.csv", index=False, header=False)
    pd.DataFrame(local_views).to_csv("batch_output/batch_local_views.csv", index=False, header=False)

    print("\n✅ Batch processing complete.")
    print("Files saved in 'batch_output/' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch preprocess light curves into global/local views.")
    parser.add_argument("--batch", type=str, help="Path to CSV file containing multiple targets.")
    args = parser.parse_args()

    if args.batch:
        process_batch(args.batch)
    else:
        print("Usage: python preprocess_data.py --batch targets.csv")