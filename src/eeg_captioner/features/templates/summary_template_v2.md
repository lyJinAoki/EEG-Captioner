# **Segment-level EEG Numeric Summary**

## **Quality control**

- bad-channel ratio: {bad_channel_ratio}
- valid channels: {valid_channels}
- bad channels: {bad_channels}

## **Scope**
- temporal structure: early, middle, last
- change rule:
  - middle_vs_early = middle value - early value
  - last_vs_middle = last value - middle value
- missing-value rule: if either value is invalid, bad-channel-related, or NaN, the corresponding change value is NaN.

---

# **Part 1: Early Feature Values**

{early_segment_eeg_feature_summaryv1}

---

# **Part 2: Middle Feature Values**

{middle_segment_eeg_feature_summaryv1}

---

# **Part 3: Last Feature Values**

{last_segment_eeg_feature_summaryv1}

---

# **Part 4: Feature Change Values**

## **Bandpower changes**
### **Per-channel bandpower changes**
| Channel | Band | Δabs power middle-early µV² | Δrel power middle-early | Δtotal power middle-early µV² | Δabs power last-middle µV² | Δrel power last-middle | Δtotal power last-middle µV² |
|---|---|---:|---:|---:|---:|---:|---:|
| {channel} | delta | {delta_abs_me} | {delta_rel_me} | {total_power_me} | {delta_abs_lm} | {delta_rel_lm} | {total_power_lm} |
| {channel} | theta | {theta_abs_me} | {theta_rel_me} | {total_power_me} | {theta_abs_lm} | {theta_rel_lm} | {total_power_lm} |
| {channel} | alpha | {alpha_abs_me} | {alpha_rel_me} | {total_power_me} | {alpha_abs_lm} | {alpha_rel_lm} | {total_power_lm} |
| {channel} | beta | {beta_abs_me} | {beta_rel_me} | {total_power_me} | {beta_abs_lm} | {beta_rel_lm} | {total_power_lm} |
| {channel} | gamma | {gamma_abs_me} | {gamma_rel_me} | {total_power_me} | {gamma_abs_lm} | {gamma_rel_lm} | {total_power_lm} |

> Repeat for all EEG channels.

### **ROI bandpower changes**
| ROI | Band | ΔROI abs power middle-early µV² | ΔROI rel power middle-early | ΔROI abs power last-middle µV² | ΔROI rel power last-middle |
|---|---|---:|---:|---:|---:|
| {roi} | delta | {roi_delta_abs_me} | {roi_delta_rel_me} | {roi_delta_abs_lm} | {roi_delta_rel_lm} |
| {roi} | theta | {roi_theta_abs_me} | {roi_theta_rel_me} | {roi_theta_abs_lm} | {roi_theta_rel_lm} |
| {roi} | alpha | {roi_alpha_abs_me} | {roi_alpha_rel_me} | {roi_alpha_abs_lm} | {roi_alpha_rel_lm} |
| {roi} | beta | {roi_beta_abs_me} | {roi_beta_rel_me} | {roi_beta_abs_lm} | {roi_beta_rel_lm} |
| {roi} | gamma | {roi_gamma_abs_me} | {roi_gamma_rel_me} | {roi_gamma_abs_lm} | {roi_gamma_rel_lm} |

---

## **Advanced spectral feature changes**
| Channel | Δtheta/beta middle-early | Δalpha/theta middle-early | ΔIAF middle-early Hz | Δspectral entropy middle-early | ΔSEF middle-early Hz | Δaperiodic exponent middle-early | Δalpha oscillatory power middle-early µV² | Δtheta/beta last-middle | Δalpha/theta last-middle | ΔIAF last-middle Hz | Δspectral entropy last-middle | ΔSEF last-middle Hz | Δaperiodic exponent last-middle | Δalpha oscillatory power last-middle µV² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| {channel} | {dtb_me} | {dat_me} | {diaf_me} | {dentropy_me} | {dsef_me} | {dexp_me} | {dalphaosc_me} | {dtb_lm} | {dat_lm} | {diaf_lm} | {dentropy_lm} | {dsef_lm} | {dexp_lm} | {dalphaosc_lm} |

> Repeat for all EEG channels.

---

## **Spatial asymmetry changes**

| Pair | Band | Δasymmetry log middle-early | Δlateralization index middle-early | Δasymmetry log last-middle | Δlateralization index last-middle |
|---|---|---:|---:|---:|---:|
| {pair} | delta | {dasym_delta_me} | {dlat_delta_me} | {dasym_delta_lm} | {dlat_delta_lm} |
| {pair} | theta | {dasym_theta_me} | {dlat_theta_me} | {dasym_theta_lm} | {dlat_theta_lm} |
| {pair} | alpha | {dasym_alpha_me} | {dlat_alpha_me} | {dasym_alpha_lm} | {dlat_alpha_lm} |
| {pair} | beta | {dasym_beta_me} | {dlat_beta_me} | {dasym_beta_lm} | {dlat_beta_lm} |
| {pair} | gamma | {dasym_gamma_me} | {dlat_gamma_me} | {dasym_gamma_lm} | {dlat_gamma_lm} |

---

## **Spatial ROI ratio changes**

| ROI_Pair | TimeFeature | delta | theta | alpha | beta | gamma |
|----------|-------------|-------|-------|-------|------|-------|
| F/C | middle-early | {delta_middle_early_F_C} | {theta_middle_early_F_C} | {alpha_middle_early_F_C} | {beta_middle_early_F_C} | {gamma_middle_early_F_C} |
| F/C | last-middle | {delta_last_middle_F_C} | {theta_last_middle_F_C} | {alpha_last_middle_F_C} | {beta_last_middle_F_C} | {gamma_last_middle_F_C} |
| F/P | middle-early | {delta_middle_early_F_P} | {theta_middle_early_F_P} | {alpha_middle_early_F_P} | {beta_middle_early_F_P} | {gamma_middle_early_F_P} |
| F/P | last-middle | {delta_last_middle_F_P} | {theta_last_middle_F_P} | {alpha_last_middle_F_P} | {beta_last_middle_F_P} | {gamma_last_middle_F_P} |
| F/T | middle-early | {delta_middle_early_F_T} | {theta_middle_early_F_T} | {alpha_middle_early_F_T} | {beta_middle_early_F_T} | {gamma_middle_early_F_T} |
| F/T | last-middle | {delta_last_middle_F_T} | {theta_last_middle_F_T} | {alpha_last_middle_F_T} | {beta_last_middle_F_T} | {gamma_last_middle_F_T} |
| F/O | middle-early | {delta_middle_early_F_O} | {theta_middle_early_F_O} | {alpha_middle_early_F_O} | {beta_middle_early_F_O} | {gamma_middle_early_F_O} |
| F/O | last-middle | {delta_last_middle_F_O} | {theta_last_middle_F_O} | {alpha_last_middle_F_O} | {beta_last_middle_F_O} | {gamma_last_middle_F_O} |
| C/P | middle-early | {delta_middle_early_C_P} | {theta_middle_early_C_P} | {alpha_middle_early_C_P} | {beta_middle_early_C_P} | {gamma_middle_early_C_P} |
| C/P | last-middle | {delta_last_middle_C_P} | {theta_last_middle_C_P} | {alpha_last_middle_C_P} | {beta_last_middle_C_P} | {gamma_last_middle_C_P} |
| C/T | middle-early | {delta_middle_early_C_T} | {theta_middle_early_C_T} | {alpha_middle_early_C_T} | {beta_middle_early_C_T} | {gamma_middle_early_C_T} |
| C/T | last-middle | {delta_last_middle_C_T} | {theta_last_middle_C_T} | {alpha_last_middle_C_T} | {beta_last_middle_C_T} | {gamma_last_middle_C_T} |
| C/O | middle-early | {delta_middle_early_C_O} | {theta_middle_early_C_O} | {alpha_middle_early_C_O} | {beta_middle_early_C_O} | {gamma_middle_early_C_O} |
| C/O | last-middle | {delta_last_middle_C_O} | {theta_last_middle_C_O} | {alpha_last_middle_C_O} | {beta_last_middle_C_O} | {gamma_last_middle_C_O} |
| P/T | middle-early | {delta_middle_early_P_T} | {theta_middle_early_P_T} | {alpha_middle_early_P_T} | {beta_middle_early_P_T} | {gamma_middle_early_P_T} |
| P/T | last-middle | {delta_last_middle_P_T} | {theta_last_middle_P_T} | {alpha_last_middle_P_T} | {beta_last_middle_P_T} | {gamma_last_middle_P_T} |
| P/O | middle-early | {delta_middle_early_P_O} | {theta_middle_early_P_O} | {alpha_middle_early_P_O} | {beta_middle_early_P_O} | {gamma_middle_early_P_O} |
| P/O | last-middle | {delta_last_middle_P_O} | {theta_last_middle_P_O} | {alpha_last_middle_P_O} | {beta_last_middle_P_O} | {gamma_last_middle_P_O} |
| T/O | middle-early | {delta_middle_early_T_O} | {theta_middle_early_T_O} | {alpha_middle_early_T_O} | {beta_middle_early_T_O} | {gamma_middle_early_T_O} |
| T/O | last-middle | {delta_last_middle_T_O} | {theta_middle_early_T_O} | {alpha_last_middle_T_O} | {beta_last_middle_T_O} | {gamma_last_middle_T_O} |

---

## **Source-network feature changes**
| Network | Band | Δsource abs power middle-early Am² | Δsource rel power middle-early | Δsource total power middle-early Am² | Δsource abs power last-middle Am² | Δsource rel power last-middle | Δsource total power last-middle Am² |
|---|---|---:|---:|---:|---:|---:|---:|
| {network} | delta | {src_delta_abs_me} | {src_delta_rel_me} | {src_total_me} | {src_delta_abs_lm} | {src_delta_rel_lm} | {src_total_lm} |
| {network} | theta | {src_theta_abs_me} | {src_theta_rel_me} | {src_total_me} | {src_theta_abs_lm} | {src_theta_rel_lm} | {src_total_lm} |
| {network} | alpha | {src_alpha_abs_me} | {src_alpha_rel_me} | {src_total_me} | {src_alpha_abs_lm} | {src_alpha_rel_lm} | {src_total_lm} |
| {network} | beta | {src_beta_abs_me} | {src_beta_rel_me} | {src_total_me} | {src_beta_abs_lm} | {src_beta_rel_lm} | {src_total_lm} |
| {network} | gamma | {src_gamma_abs_me} | {src_gamma_rel_me} | {src_total_me} | {src_gamma_abs_lm} | {src_gamma_rel_lm} | {src_total_lm} |

---

## **Time-domain feature changes**

| Channel | Δmean middle-early µV | Δstd middle-early µV | ΔRMS middle-early µV | Δpeak-to-peak middle-early µV | Δzero-crossing middle-early | Δline length middle-early µV | ΔHjorth mobility middle-early | ΔHjorth complexity middle-early | Δmean last-middle µV | Δstd last-middle µV | ΔRMS last-middle µV | Δpeak-to-peak last-middle µV | Δzero-crossing last-middle | Δline length last-middle µV | ΔHjorth mobility last-middle | ΔHjorth complexity last-middle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| {channel} | {dmean_me} | {dstd_me} | {drms_me} | {dp2p_me} | {dzcr_me} | {dll_me} | {dmob_me} | {dcomp_me} | {dmean_lm} | {dstd_lm} | {drms_lm} | {dp2p_lm} | {dzcr_lm} | {dll_lm} | {dmob_lm} | {dcomp_lm} |
