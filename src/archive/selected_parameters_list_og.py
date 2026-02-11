# Selected Gaia DR3 parameters for XGBoost classifier
# 53 Gaia columns queried from DB + derived features computed in Python

import numpy as np

# Columns to query from gaia_dr3.gaia_source (used in SQL)
GAIA_QUERY_COLUMNS = [
    'ra_error',
    'dec_error',
    'parallax',
    'parallax_error',
    'parallax_over_error',
    'pm',
    'pmra',
    'pmra_error',
    'pmdec',
    'pmdec_error',
    'astrometric_n_obs_al',
    'astrometric_n_obs_ac',
    'astrometric_n_good_obs_al',
    'astrometric_n_bad_obs_al',
    'astrometric_gof_al',
    'astrometric_chi2_al',
    'astrometric_excess_noise',
    'astrometric_excess_noise_sig',
    'astrometric_params_solved',
    'astrometric_matched_transits',
    'visibility_periods_used',
    'astrometric_sigma5d_max',
    'matched_transits',
    'ipd_gof_harmonic_amplitude',
    'ipd_gof_harmonic_phase',
    'ipd_frac_multi_peak',
    'ipd_frac_odd_win',
    'ruwe',
    'duplicated_source',
    'phot_g_n_obs',
    'phot_g_mean_flux',
    'phot_g_mean_flux_error',
    'phot_g_mean_flux_over_error',
    'phot_g_mean_mag',
    'phot_bp_n_obs',
    'phot_bp_mean_flux',
    'phot_bp_mean_flux_error',
    'phot_bp_mean_flux_over_error',
    'phot_bp_mean_mag',
    'phot_rp_n_obs',
    'phot_rp_mean_flux',
    'phot_rp_mean_flux_error',
    'phot_rp_mean_flux_over_error',
    'phot_rp_mean_mag',
    'phot_bp_rp_excess_factor',
    'phot_bp_n_contaminated_transits',
    'phot_bp_n_blended_transits',
    'phot_rp_n_contaminated_transits',
    'phot_rp_n_blended_transits',
    'phot_proc_mode',
    'bp_rp',
    'bp_g',
    'g_rp',
]


# --- Derived features (computed from the above Gaia columns) ---

DERIVED_FEATURES = [
    'corrected_bp_rp_excess',
]


def compute_derived_features(df):
    """
    Compute derived features from existing Gaia columns and add them to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'phot_bp_rp_excess_factor' and 'bp_rp' columns.

    Returns
    -------
    df : pd.DataFrame
        Same DataFrame with derived feature columns added.
    """
    # Corrected BP/RP flux excess factor C*
    # From Riello et al. 2021, Table 2 (Gaia EDR3 photometric validation)
    # C* = C - f(G_BP - G_RP), where f is a piecewise polynomial
    C = df['phot_bp_rp_excess_factor'].astype(float)
    x = df['bp_rp'].astype(float)
    f = np.where(x < 0.5,
                 1.154360 + 0.033772 * x + 0.032277 * x**2,
                 np.where(x < 4.0,
                          1.162004 + 0.011464 * x + 0.049255 * x**2 - 0.005879 * x**3,
                          1.057572 + 0.140537 * x))
    df['corrected_bp_rp_excess'] = C - f

    return df


# Full feature list for training: Gaia columns + derived
# Exported as SELECTED_GAIA_PARAMETERS so existing scripts don't need changes
SELECTED_GAIA_PARAMETERS = GAIA_QUERY_COLUMNS + DERIVED_FEATURES
