import argparse
import numpy as np
import pandas as pd
import lightkurve as lk

# NOTE: This is a simplified version of the preprocessing logic for a standalone script.
# It should mirror the core logic in your cnn_preprocessing.py file.

def process_light_curve(target_id, period, t0, duration, mission):
    """
    Downloads, flattens, folds, and bins a light curve for a single target.
    """
    print(f"Searching for {target_id} in {mission} data...")
    search_result = lk.search_lightcurve(target_id, mission=mission)
    
    if not search_result:
        raise ValueError(f"No light curve found for target {target_id} in mission {mission}.")
        
    lc_collection = search_result.download_all()
    lc = lc_collection.stitch().remove_nans().remove_outliers()
    
    # Flatten, fold, and bin the light curve
    folded_lc = lc.fold(period=period, epoch_time=t0)
    
    # --- Create Global View (2001 bins) ---
    binned_global_lc = folded_lc.bin(time_bin_size=period / 2001)
    global_view = binned_global_lc.flux.value
    # Normalize the global view
    global_view = (global_view - np.median(global_view)) / (np.median(global_view) - np.min(global_view))
    
    # --- Create Local View (201 bins) ---
    phase_mask = (folded_lc.time.value > -0.1) & (folded_lc.time.value < 0.1)
    local_folded_lc = folded_lc[phase_mask]
    binned_local_lc = local_folded_lc.bin(time_bin_size=0.001, n_bins=201)
    local_view = binned_local_lc.flux.value
    # Normalize the local view
    local_view = (local_view - np.median(local_view)) / (np.median(local_view) - np.min(local_view))
    
    # Ensure views are the correct length, padding if necessary
    global_view_final = np.pad(global_view, (0, 2001 - len(global_view)), 'constant', constant_values=0)
    local_view_final = np.pad(local_view, (0, 201 - len(local_view)), 'constant', constant_values=0)

    return global_view_final, local_view_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess exoplanet light curve data.")
    parser.add_argument("--target_id", type=str, required=True, help="Target ID (e.g., 'KIC 11442793')")
    parser.add_argument("--period", type=float, required=True, help="Orbital period in days.")
    parser.add_argument("--t0", type=float, required=True, help="Transit time (epoch) in BJD.")
    parser.add_argument("--duration", type=float, required=True, help="Transit duration in hours.")
    parser.add_argument("--mission", type=str, default="Kepler", help="Mission (Kepler, K2, or TESS).")
    
    args = parser.parse_args()
    
    print("Starting preprocessing...")
    try:
        global_view, local_view = process_light_curve(
            args.target_id, args.period, args.t0, args.duration, args.mission
        )
        
        # Save to CSV files
        global_df = pd.DataFrame([global_view])
        local_df = pd.DataFrame([local_view])
        
        global_filename = f"{args.target_id.replace(' ', '_')}_global_view.csv"
        local_filename = f"{args.target_id.replace(' ', '_')}_local_view.csv"
        
        global_df.to_csv(global_filename, index=False)
        local_df.to_csv(local_filename, index=False)
        
        print(f"\nSuccessfully created files:")
        print(f"- {global_filename}")
        print(f"- {local_filename}")
        print("\nYou can now upload these files to the Research Platform.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your input parameters and ensure required libraries (lightkurve, pandas) are installed.")