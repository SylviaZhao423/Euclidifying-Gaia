#!/usr/bin/env python3
"""
Gaia (BP, G, RP) + Euclid VIS bandpass comparison plot.

Reproduces the style of Jordi et al. 2010 (A&A) and overlays the
Euclid VIS band to show where it sits relative to Gaia's passbands.

Filter data sourced from the SVO Filter Profile Service:
  Gaia DR3: GAIA/GAIA3.G, GAIA/GAIA3.Gbp, GAIA/GAIA3.Grp
  Euclid:   Euclid/VIS.vis

Usage:
    python src/analyze_bandpass.py

Output:
    output/bandpass_gaia_euclid.png
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_DIR = Path(__file__).resolve().parents[1]
FILTER_DIR  = PROJECT_DIR / "data" / "filters"
OUTPUT_FILE = PROJECT_DIR / "output" / "bandpass_gaia_euclid.png"


def load_svo_filter(path):
    """Parse SVO VOTable XML → (wavelength_nm, transmission) arrays."""
    tree = ET.parse(path)
    root = tree.getroot()
    # Strip namespace for easier searching
    ns = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
    prefix = f"{{{ns}}}" if ns else ""

    values = []
    for tr in root.iter(f"{prefix}TR"):
        tds = list(tr.iter(f"{prefix}TD"))
        if len(tds) >= 2:
            wl = float(tds[0].text.strip())
            tx = float(tds[1].text.strip())
            values.append((wl, tx))

    arr = np.array(values)
    wl_nm = arr[:, 0] / 10.0   # Å → nm
    tx    = arr[:, 1]
    # Clip numerical noise
    tx = np.where(tx < 1e-4, 0.0, tx)
    # Normalise to peak = 1
    tx = tx / tx.max()
    return wl_nm, tx


def main():
    gaia_G_wl,  gaia_G_tx  = load_svo_filter(FILTER_DIR / "gaia_G.xml")
    gaia_BP_wl, gaia_BP_tx = load_svo_filter(FILTER_DIR / "gaia_BP.xml")
    gaia_RP_wl, gaia_RP_tx = load_svo_filter(FILTER_DIR / "gaia_RP.xml")
    euclid_wl,  euclid_tx  = load_svo_filter(FILTER_DIR / "euclid_VIS.xml")

    # Euclid VIS: zero out the pre-filter leak below ~530 nm
    euclid_tx[euclid_wl < 530] = 0.0
    euclid_tx = euclid_tx / euclid_tx.max()

    fig, ax = plt.subplots(figsize=(8, 4.3))

    # Gaia bands — matching Jordi+2010 style
    ax.plot(gaia_BP_wl, gaia_BP_tx, color="royalblue",  lw=2.0, ls=":",
            label=r"$G_{\rm BP}$")
    ax.plot(gaia_G_wl,  gaia_G_tx,  color="black",       lw=2.2, ls="-",
            label=r"$G$")
    ax.plot(gaia_RP_wl, gaia_RP_tx, color="tomato",      lw=2.0, ls="--",
            label=r"$G_{\rm RP}$")

    # Euclid VIS — filled + outline
    ax.fill_between(euclid_wl, euclid_tx, alpha=0.18, color="darkorchid")
    ax.plot(euclid_wl, euclid_tx, color="darkorchid", lw=2.4, ls="-",
            label=r"Euclid $I_{\rm E}$")

    ax.set_xlabel(r"wavelength (nm)", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_ylabel("Normalised response (energy)", fontsize=15, fontweight="bold", labelpad=15)
    ax.set_xlim(300, 1100)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=14, loc="upper right", framealpha=0.7)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=13)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
