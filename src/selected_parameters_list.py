# Selected Gaia DR3 parameters for XGBoost classifier
# See GAIA_QUERY_COLUMNS at the bottom for the combined list.

# =============================================================================
# Original Gaia DR3 columns
# =============================================================================
GAIA_ORIGINAL_COLUMNS = [
    ### Astrometry errors =====================================================
    # 'ra_error',
    # 'dec_error',
    'parallax_over_error',
    ### Astrometric quality ===================================================
    'astrometric_n_obs_al',
    # 'astrometric_n_obs_ac',
    'astrometric_n_good_obs_al',
    'astrometric_n_bad_obs_al',
    'astrometric_gof_al',
    'astrometric_chi2_al',
    # 'astrometric_excess_noise',
    'astrometric_excess_noise_sig',
    # 'astrometric_params_solved',
    'astrometric_matched_transits',
    # 'visibility_periods_used',
    # 'astrometric_sigma5d_max',
    'matched_transits',
    ### Image parameters =======================================================
    'ipd_gof_harmonic_amplitude',
    'ipd_gof_harmonic_phase',
    # 'ipd_frac_multi_peak',
    # 'ipd_frac_odd_win',
    'ruwe',
    # 'duplicated_source',
    ### G-band photometry ======================================================
    # 'phot_g_n_obs',
    # 'phot_g_mean_flux',
    # 'phot_g_mean_flux_error',
    'phot_g_mean_flux_over_error',
    'phot_g_mean_mag',
    ### BP-band photometry =====================================================
    # 'phot_bp_n_obs',
    # phot_bp_mean_flux — redundant with phot_g_mean_flux + bp_g color
    # 'phot_bp_mean_flux_error',
    'phot_bp_mean_flux_over_error',
    # phot_bp_mean_mag — redundant with phot_g_mean_mag + bp_g color
    ### RP-band photometry =====================================================
    # 'phot_rp_n_obs',
    # phot_rp_mean_flux — redundant with phot_g_mean_flux + g_rp color
    # 'phot_rp_mean_flux_error',
    'phot_rp_mean_flux_over_error',
    # phot_rp_mean_mag — redundant with phot_g_mean_mag + g_rp color
    ### BP/RP quality (could try later)==========================================================
    # 'phot_bp_n_contaminated_transits',
    # 'phot_bp_n_blended_transits',
    # 'phot_rp_n_contaminated_transits',
    # 'phot_rp_n_blended_transits',
    # 'phot_proc_mode',
    ### Colors ================================================================
    # 'bp_rp',
    'bp_g',
    'g_rp',
]

# =============================================================================
# Derived columns (pre-computed in WSDB via add_derived_columns.py)
# =============================================================================
DERIVED_COLUMNS = [
    'corrected_bp_rp_excess',  # C* from Riello et al. 2021, Table 2
    'pm_significance',         # sqrt(chi2) with pmra_pmdec_corr
    'position_error',     # sqrt(det(C)) with ra_dec_corr
]

# =============================================================================
# Combined list for training
# =============================================================================
GAIA_QUERY_COLUMNS = GAIA_ORIGINAL_COLUMNS + DERIVED_COLUMNS
SELECTED_GAIA_PARAMETERS = GAIA_QUERY_COLUMNS
N_FEATURES = len(SELECTED_GAIA_PARAMETERS)
