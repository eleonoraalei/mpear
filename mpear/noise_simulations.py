import numpy as np
import pandas as pd
import io
from contextlib import redirect_stdout, redirect_stderr

from pyEDITH.astrophysical_scene import calc_flux_zero_point
from pyEDITH import calculate_texp, calculate_snr
from pyEDITH.units import *

from .flux_calculations import *


def primary_bandpass_detection_exptime(
    parameters, wavelength, stellar_flux, planet_contrast
):
    """
    Calculate the exposure time for primary bandpass detection.

    Args:
        parameters (dict): Dictionary of input parameters.
        wavelength (array): Array of wavelengths.
        stellar_flux (array): Array of stellar flux values.
        planet_contrast (array): Array of planet contrast values.

    Returns:
        tuple: Exposure time and updated parameters.
    """
    # Set bandwidth for discovery
    # Calculate magnitudes at V band (0.55 micron)
    lam_vband = np.argmin(np.abs(wavelength - 0.55))
    parameters["FstarV_10pc"] = stellar_flux[lam_vband].value
    # Calculate magnitudes at 0.5 micron
    lam_min_primary, lam_max_primary, deltalambda_primary = calculate_deltalambda_edges(
        parameters["wavelength"][0],
        parameters["bandwidth"],
        parameters["resolution"][0],
    )
    parameters["Fstar_10pc"] = [
        find_average_fluxes(
            wavelength, stellar_flux, [lam_min_primary, lam_max_primary]
        ).value
    ]
    parameters["Fp/Fs"] = [
        find_average_fluxes(
            wavelength, planet_contrast, [lam_min_primary, lam_max_primary]
        )
    ]

    # Calculate exposure time
    texp, _ = calculate_texp(parameters, verbose=False)
    print("Reference exposure time: ", texp.to(u.hr))
    return texp, parameters


def secondary_bandpass_scanning(
    band,
    texp,
    wavelength,
    stellar_flux,
    planet_contrast,
    parameters,
    secondary_parameters,
    scan_number=50,
):
    """
    Perform secondary bandpass scanning.

    Args:
        band (float): Bandwidth value.
        texp (float): Exposure time.
        wavelength (array): Array of wavelengths.
        stellar_flux (array): Array of stellar flux values.
        planet_contrast (array): Array of planet contrast values.
        parameters (dict): Dictionary of primary parameters for pyEDITH.
        secondary_parameters (dict): Dictionary of secondary parameters for pyEDITH.
        scan_number (int): Number of scan points (default: 50).

    Returns:
        tuple: Observed Earth data, observed star data, and count rates.
    """
    secondary_parameters["bandwidth"] = band
    observed_earth = pd.DataFrame(columns=["wavelength", "Fp/Fs", "snr", "bandwidth"])
    observed_star = pd.DataFrame(columns=["wavelength", "flux", "bandwidth"])
    cr_list = []

    # Determine wavelength range for scanning
    if scan_number == "full":
        lam_range = wavelength
    else:
        lam_range = np.linspace(0.3, 1.75, scan_number)

    # Perform scanning for each wavelength
    for lam in lam_range:
        # Calculate secondary parameters
        secondary_parameters["wavelength"] = [lam]
        secondary_parameters["FstarV_10pc"] = parameters["FstarV_10pc"]
        lam_min_secondary, lam_max_secondary, deltalambda_secondary = (
            calculate_deltalambda_edges(lam, band, parameters["resolution"][0])
        )
        secondary_parameters["Fstar_10pc"] = [
            find_average_fluxes(
                wavelength, stellar_flux, [lam_min_secondary, lam_max_secondary]
            ).value
        ]
        secondary_parameters["Fp/Fs"] = [
            find_average_fluxes(
                wavelength, planet_contrast, [lam_min_secondary, lam_max_secondary]
            )
        ]
        # Calculate SNR
        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            snr, validation_variables = calculate_snr(
                secondary_parameters, texp, verbose=False
            )
        # Save flux and SNR data
        observed_earth.loc[len(observed_earth)] = [
            lam,
            secondary_parameters["Fp/Fs"][0],
            snr[0],
            deltalambda_secondary,
        ]
        observed_star.loc[len(observed_star)] = [
            lam,
            secondary_parameters["Fstar_10pc"][0],
            deltalambda_secondary,
        ]
        # Save count rates
        try:
            cr = {"wavelength": lam}
            for col in [
                "CRp",
                "CRnf",
                "CRbs",
                "CRbz",
                "CRbez",
                "CRbbin",
                "CRbd",
                "CRbth",
                "CRb",
            ]:
                cr[col] = validation_variables[0][col].value
            cr_list.append(cr)
        except KeyError:
            print("Could not save Count Rates for lambda: " + str(lam))

    # Convert data types
    observed_earth["wavelength"] = pd.to_numeric(
        observed_earth["wavelength"], errors="coerce"
    )
    observed_earth["Fp/Fs"] = pd.to_numeric(observed_earth["Fp/Fs"], errors="coerce")
    observed_earth["bandwidth"] = pd.to_numeric(
        observed_earth["bandwidth"], errors="coerce"
    )

    observed_star["wavelength"] = pd.to_numeric(
        observed_star["wavelength"], errors="coerce"
    )
    observed_star["flux"] = pd.to_numeric(observed_star["flux"], errors="coerce")
    observed_star["bandwidth"] = pd.to_numeric(
        observed_star["bandwidth"], errors="coerce"
    )

    # Create DataFrame for count rates
    countrates = pd.DataFrame(cr_list)
    countrates.set_index("wavelength", inplace=True)
    countrates = countrates.select_dtypes(include=[np.number])

    return observed_earth, observed_star, countrates


def sweep_bandpasses(
    flux_file,
    parameters,
    secondary_parameters,
    bandwidths,
    scan_number=50,
    savefolder=None,
):
    """
    Perform bandpass sweeping across the wavelength range for every user-specified bandwidth.

    Args:
        flux_file (str): Path to the flux file.
        parameters (dict): Dictionary of primary parameters for pyEDITH.
        secondary_parameters (dict): Dictionary of secondary parameters for pyEDITH.
        bandwidths (list): List of bandwidth values to sweep.
        scan_number (int): Number of scan points (default: 50).
        savefolder (str): Folder to save results (default: None). If none, it won't save.

    Returns:
        tuple: Observed Earth data, count rates, CRp list, and exposure time.
    """
    Crp_list = []

    # Load flux data
    flux = np.loadtxt(flux_file)
    wavelength = flux[:, 0]
    stellar_flux = flux[:, 2] * u.W / u.m**2 / u.um
    stellar_flux = stellar_flux.to(
        u.photon / (u.cm**2 * u.s * u.nm),
        equivalencies=u.spectral_density(flux[:, 0] * u.micron),
    )
    planet_contrast = flux[:, -1] / flux[:, 2]
    planet_flux = flux[:, -1] * u.W / u.m**2 / u.um
    planet_flux = planet_flux.to(
        u.photon / (u.cm**2 * u.s * u.nm),
        equivalencies=u.spectral_density(flux[:, 0] * u.micron),
    )

    # Calculate exposure time for primary bandpass detection
    texp, parameters = primary_bandpass_detection_exptime(
        parameters, wavelength, stellar_flux, planet_contrast
    )
    observed_earth_bandwidths = {}
    count_rates_dict = {}

    # Perform secondary bandpass scanning for each bandwidth
    for band in bandwidths:
        observed_earth, observed_star, count_rates = secondary_bandpass_scanning(
            band,
            texp,
            wavelength,
            stellar_flux,
            planet_contrast,
            parameters,
            secondary_parameters,
            scan_number,
        )
        observed_earth_bandwidths[band] = observed_earth
        observed_star["flux_converted"] = observed_star.apply(
            lambda row: convert_flux(row["flux"], row["wavelength"]), axis=1
        )

        observed_earth["flux"] = observed_earth["Fp/Fs"] * observed_star["flux"]
        observed_earth["flux"] = pd.to_numeric(observed_earth["flux"], errors="coerce")

        # Save results if savefolder is provided
        if savefolder is not None:
            observed_earth.to_csv(
                f"{savefolder}observed_{band}_scan" + str(scan_number) + ".csv",
                index=None,
            )
        observed_earth_bandwidths[band] = observed_earth
        count_rates_dict[band] = count_rates

        Crp_list.append(count_rates["CRp"])

    return observed_earth_bandwidths, count_rates_dict, Crp_list, texp
