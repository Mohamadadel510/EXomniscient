import argparse
import numpy as np
import pandas as pd
import lightkurve as lk
from scipy import interpolate
import warnings
import time

# --- Suppress Warnings ---
warnings.filterwarnings('ignore', category=lk.LightkurveWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Robust Preprocessing Class (from the main application) ---
class KeplerPreprocessor:
    """
    Robust preprocessing pipeline for Kepler/TESS light curves following Shallue & Vanderburg (2018).
    """
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
                if not lc_collection: return None
                return lc_collection.stitch().remove_nans()
            except Exception as e:
                print(f"-> [Attempt {attempt}/{self.max_retries}] Download for {target_id} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        return None

    def flatten_lightcurve(self, lc):
        """
        Flattens the light curve using a basis spline, with a robust fallback method.
        """
        try:
            time, flux = lc.time.value, lc.flux.value
            phase = ((time - self.t0 + 0.5 * self.period) % self.period) - 0.5 * self.period
            in_transit = np.abs(phase) < self.duration_days
            best_bic, best_spline = np.inf, None
            spacings = [0.5, 0.75, 1.0, 1.5, 2.0]
            for spacing in spacings:
                n_knots = int(np.ceil((time[-1] - time[0]) / spacing))
                if n_knots < 4: continue
                knots = np.linspace(time[0], time[-1], n_knots)
                mask = ~in_transit
                spline = None
                for _ in range(3):
                    try:
                        if np.sum(mask) <= 3: break
                        spline = interpolate.LSQUnivariateSpline(time[mask], flux[mask], knots[1:-1], k=3)
                        residuals = flux[mask] - spline(time[mask])
                        outliers = np.abs(residuals) > 3 * np.std(residuals)
                        mask[np.where(mask)[0][outliers]] = False
                    except (ValueError, TypeError, IndexError):
                        spline = None; break
                if spline:
                    residuals = flux[mask] - spline(time[mask])
                    n, k = len(time[mask]), len(knots)
                    bic = n * np.log(np.sum(residuals**2) / n) + k * np.log(n)
                    if bic < best_bic:
                        best_bic, best_spline = bic, spline
            if best_spline is None: raise ValueError("Spline fitting failed")
            return flux / best_spline(time), time
        except Exception as e:
            warnings.warn(f"-> Primary spline fit failed ('{e}'). Using robust fallback flatten.")
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
        if min_val < -1e-9: arr /= -min_val
        return arr

    def preprocess(self, lc):
        """Full preprocessing pipeline for a single light curve."""
        if lc is None: raise ValueError("LightCurve object is None.")
        flux, time = self.flatten_lightcurve(lc)
        global_view = self.fold_and_bin(time, flux, n_bins=2001, view_span_days=self.period)
        local_view = self.fold_and_bin(time, flux, n_bins=201, view_span_days=8 * self.duration_days)
        return self.normalize(global_view), self.normalize(local_view)

def process_single_target(target_id, period, t0, duration, mission):
    """Wrapper function to download and preprocess a single target."""
    print(f"\nProcessing {target_id} ({mission})...")
    preprocessor = KeplerPreprocessor(period, t0, duration, mission)
    lc = preprocessor.download_lightcurve(target_id)
    return preprocessor.preprocess(lc)

# --- Main Script Functions ---
def build_evaluation_dataset():
    """
    Downloads and processes a predefined list of targets to create an evaluation dataset.
    """
    # Confirmed exoplanets from different missions
    eval_targets = [
        # Target ID, Mission, Period (days), t0 (BJD), Duration (hours)
        ("Kepler-10 b", "Kepler", 0.837, 2454964.57, 1.8),
        ("K2-18 b", "K2", 32.94, 2457024.42, 5.5),
        ("TOI-700 d", "TESS", 37.42, 2458331.25, 3.8),
        ("Kepler-186 f", "Kepler", 129.9, 2456276.35, 4.5),
    ]

    metadata_records = []
    global_views, local_views = [], []

    for target_id, mission, period, t0, duration in eval_targets:
        try:
            global_view, local_view = process_single_target(target_id, period, t0, duration, mission)
            global_views.append(global_view)
            local_views.append(local_view)
            metadata_records.append({
                "target_id": target_id, "mission": mission, "period_days": period,
                "t0_bjd": t0, "duration_hours": duration
            })
            print(f"-> Successfully processed {target_id}.")
        except Exception as e:
            print(f"-> ⚠️ Skipping {target_id} due to error: {e}")

    if metadata_records:
        pd.DataFrame(metadata_records).to_csv("evaluation_metadata.csv", index=False)
        pd.DataFrame(global_views).to_csv("evaluation_global_views.csv", index=False, header=False)
        pd.DataFrame(local_views).to_csv("evaluation_local_views.csv", index=False, header=False)
        print("\n✅ Evaluation dataset created successfully!")
        print("   - evaluation_metadata.csv")
        print("   - evaluation_global_views.csv")
        print("   - evaluation_local_views.csv")
    else:
        print("\n❌ No valid targets were processed. Dataset not created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess exoplanet light curve data into global and local views.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Download and preprocess a predefined evaluation dataset of known exoplanets."
    )
    parser.add_argument("--target_id", type=str, help="Target ID (e.g., 'Kepler-10 b')")
    parser.add_argument("--period", type=float, help="Orbital period in days.")
    parser.add_argument("--t0", type=float, help="Transit time (epoch) in BJD (e.g., 2454964.57).")
    parser.add_argument("--duration", type=float, help="Transit duration in hours.")
    parser.add_argument("--mission", type=str, choices=['Kepler', 'K2', 'TESS'], help="Mission name.")
    
    args = parser.parse_args()

    # --- SCRIPT EXECUTION LOGIC ---
    if args.eval:
        build_evaluation_dataset()
    elif all([args.target_id, args.period, args.t0, args.duration, args.mission]):
        try:
            global_view, local_view = process_single_target(
                args.target_id, args.period, args.t0, args.duration, args.mission
            )
            # Save the processed views for the single target
            sanitized_id = args.target_id.replace(" ", "_")
            global_df = pd.DataFrame([global_view])
            local_df = pd.DataFrame([local_view])
            global_df.to_csv(f"{sanitized_id}_global.csv", index=False, header=False)
            local_df.to_csv(f"{sanitized_id}_local.csv", index=False, header=False)
            print(f"\n✅ Successfully created files for {args.target_id}.")
        except Exception as e:
            print(f"\n❌ An error occurred during processing: {e}")
    else:
        print("Usage Error: You must either use the '--eval' flag or provide all arguments for a single target.")
        print("For help, run: python create_dataset.py --help")

### How to Use the New Script

# Save the code above as `create_dataset.py`. You can now run it from your terminal in two ways:

# 1.  **To Create the Evaluation Dataset** (for the 4 predefined targets):
#     ```bash
#     python create_dataset.py --eval
#     ```
#     This will create `evaluation_metadata.csv`, `evaluation_global_views.csv`, and `evaluation_local_views.csv` in your folder.

# 2.  **To Process a Single, Custom Target**:
#     ```bash
#     python create_dataset.py --target_id "Kepler-10 b" --mission "Kepler" --period 0.837 --t0 2454964.57 --duration 1.8
