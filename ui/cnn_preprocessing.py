import numpy as np
import lightkurve as lk
from scipy import interpolate
import pandas as pd
import warnings
import time

# Suppress lightkurve warnings and runtime warnings
warnings.filterwarnings('ignore', category=lk.LightkurveWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class KeplerPreprocessor:
    """
    Preprocessing pipeline for Kepler/TESS light curves following Shallue & Vanderburg (2018).
    """
    def __init__(self, period, t0, duration, mission="Kepler", max_retries=3, retry_delay=5):
        self.period = period
        self.t0 = t0
        self.duration_days = duration / 24.0
        self.mission = mission
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def download_lightcurve(self, target_id):
        """Downloads and stitches light curves for a given target."""
        for attempt in range(1, self.max_retries + 1):
            try:
                search = lk.search_lightcurve(target_id, mission=self.mission, author='Kepler' if self.mission=='Kepler' else None)
                if not search:
                    warnings.warn(f"No light curves found for {target_id} ({self.mission})")
                    return None
                
                lc_collection = search.download_all()
                if not lc_collection:
                    return None
                
                return lc_collection.stitch().remove_nans()
            except Exception as e:
                print(f"[Attempt {attempt}/{self.max_retries}] Download for {target_id} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        return None
                    
    def remove_known_transits(self, lc, known_periods, known_t0s, known_durations):
        """Masks transits of other known planets."""
        mask = np.ones_like(lc.time.value, dtype=bool)
        for p, t0, d_hours in zip(known_periods, known_t0s, known_durations):
            d_days = d_hours / 24.0
            phase = ((lc.time.value - t0 + 0.5 * p) % p) - 0.5 * p
            is_in_transit = np.abs(phase) < 1.5 * d_days
            mask[is_in_transit] = False
        return lc[mask]

    def flatten_lightcurve(self, lc):
        """
        Flattens the light curve using a basis spline. If that fails, it uses
        lightkurve's robust default flatten method as a fallback.
        """
        try:
            # --- Primary Method: Basis Spline Fitting ---
            time, flux = lc.time.value, lc.flux.value
            phase = ((time - self.t0 + 0.5 * self.period) % self.period) - 0.5 * self.period
            in_transit = np.abs(phase) < self.duration_days

            best_bic, best_spline = np.inf, None
            break_point_spacings = [0.5, 0.75, 1.0, 1.5, 2.0]

            for spacing in break_point_spacings:
                n_knots = int(np.ceil((time[-1] - time[0]) / spacing))
                if n_knots < 4: continue

                knots = np.linspace(time[0], time[-1], n_knots)
                mask = ~in_transit
                spline = None
                for _ in range(3): # Iterative outlier rejection
                    try:
                        # Ensure there are enough points to fit the spline
                        if np.sum(mask) <= 3: break 
                        spline = interpolate.LSQUnivariateSpline(time[mask], flux[mask], knots[1:-1], k=3)
                        residuals = flux[mask] - spline(time[mask])
                        outliers = np.abs(residuals) > 3 * np.std(residuals)
                        mask[np.where(mask)[0][outliers]] = False
                    except (ValueError, TypeError, IndexError):
                        spline = None
                        break
                
                if spline:
                    residuals = flux[mask] - spline(time[mask])
                    n = len(time[mask])
                    k = len(knots)
                    bic = n * np.log(np.sum(residuals**2) / n) + k * np.log(n)
                    if bic < best_bic:
                        best_bic, best_spline = bic, spline
            
            if best_spline is None:
                raise ValueError("Basis spline fitting failed.")
            
            # If spline fitting was successful
            spline_model = best_spline(time)
            return flux / spline_model, time

        except Exception as e:
            # --- Fallback Method: Lightkurve's Robust Flatten ---
            warnings.warn(f"Primary spline fitting failed: '{e}'. Using robust fallback method.")
            try:
                # Use a window size that is large enough to not flatten the transit itself.
                # A window of 20x the transit duration is a safe choice.
                cadence = np.median(np.diff(lc.time.value)) # Time between observations
                window_duration_in_cadences = int(20 * self.duration_days / cadence)
                
                # window_length must be an odd integer
                if window_duration_in_cadences % 2 == 0:
                    window_duration_in_cadences += 1
                
                # Set a reasonable minimum window size
                window_length = max(101, window_duration_in_cadences)

                flattened_lc = lc.flatten(window_length=window_length)
                return flattened_lc.flux.value, flattened_lc.time.value
            except Exception as fallback_e:
                raise ValueError(f"Both primary and fallback flattening methods failed. Last error: {fallback_e}")


    def fold_and_bin(self, time, flux, n_bins, view_span_days):
        """Folds the light curve and bins it."""
        phase = ((time - self.t0 + 0.5 * self.period) % self.period) / self.period
        phase[phase > 0.5] -= 1.0
        
        phase_time = phase * self.period
        bin_centers = np.linspace(-view_span_days / 2, view_span_days / 2, n_bins)
        bin_width = np.abs(bin_centers[1] - bin_centers[0])
        
        binned_flux = np.zeros(n_bins)
        for i, center in enumerate(bin_centers):
            in_bin = np.abs(phase_time - center) < bin_width / 2
            binned_flux[i] = np.median(flux[in_bin]) if np.any(in_bin) else np.nan
        
        # Interpolate to fill empty bins
        if np.any(np.isnan(binned_flux)):
            valid_mask = ~np.isnan(binned_flux)
            if np.sum(valid_mask) > 1:
                binned_flux = np.interp(bin_centers, bin_centers[valid_mask], binned_flux[valid_mask])
            else:
                binned_flux = np.nan_to_num(binned_flux, nan=1.0)
                
        return binned_flux

    def normalize(self, arr):
        """Normalizes an array by shifting median to 0 and scaling min to -1."""
        if np.all(np.isfinite(arr)) and len(arr) > 0:
            arr -= np.median(arr)
            min_val = np.min(arr)
            if min_val < -1e-9:
                arr /= -min_val
        return arr

    def preprocess(self, lc, known_periods=None, known_t0s=None, known_durations=None):
        """Full preprocessing pipeline."""
        if lc is None:
            raise ValueError("LightCurve object is None. Download may have failed.")
        
        if known_periods:
            lc = self.remove_known_transits(lc, known_periods, known_t0s, known_durations)
        
        flux, time = self.flatten_lightcurve(lc)
        
        global_view = self.fold_and_bin(time, flux, n_bins=2001, view_span_days=self.period)
        local_view = self.fold_and_bin(time, flux, n_bins=201, view_span_days=8 * self.duration_days)
        
        return self.normalize(global_view), self.normalize(local_view)

def process_single_target(target_id, period, t0, duration, mission, 
                          known_periods=None, known_t0s=None, known_durations=None):
    """Convenience function to process a single target from parameters."""
    preprocessor = KeplerPreprocessor(period=period, t0=t0, duration=duration, mission=mission)
    lc = preprocessor.download_lightcurve(target_id)
    return preprocessor.preprocess(lc, known_periods, known_t0s, known_durations)

