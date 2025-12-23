import os
import spectres
import numpy as np
import pandas as pd
from astropy import units as u
import astropy.constants as const
from .orbital_mechanics import calculate_orbit_location, calculate_apparent_phase


def mod_cfg_line(cfg, m_line, new_values):
    """
    Modify a specific line in the configuration.

    Args:
        cfg (list): List of configuration lines.
        m_line (str): Line to be modified.
        new_values (str): New values to be inserted.

    Returns:
        list: Updated configuration lines.
    """
    for c, line in enumerate(cfg):
        if m_line in line:
            cfg[c] = m_line + str(new_values) + "\n"
    return cfg


def write_config(config_out, cfg):
    """
    Write configuration to a file.

    Args:
        config_out (str): Output file path.
        cfg (list): Configuration lines to write.
    """
    try:
        os.remove(config_out)
    except:
        pass
    with open(config_out, "w") as f:
        f.writelines(cfg)


def run_psg_docker(cfg_out, rad_out):
    """
    Run PSG Docker container with given configuration.

    Args:
        cfg_out (str): Path to configuration file.
        rad_out (str): Path to output radiation file.
    """
    psgurl = "http://localhost:3000"
    os.system(
        f"curl -s -d type=rad -d wgeo=y --data-urlencode file@{cfg_out} {psgurl}/api.php > {rad_out}"
    )


def convert_flux(flux, wavelength):
    """
    Convert flux units from photons to W/m^2/um.

    Args:
        flux (float): Flux in photons/(cm^2 * s * nm).
        wavelength (float): Wavelength in microns.

    Returns:
        float: Flux in W/m^2/um.
    """
    return (
        (flux * u.photon / (u.cm**2 * u.s * u.nm))
        .to(u.W / u.m**2 / u.um, equivalencies=u.spectral_density(wavelength * u.um))
        .value
    )


def psg_calculate_and_load_flux(cfg_file, rad_file):
    """
    Calculate and load flux using PSG.

    Args:
        cfg_file (str): Path to configuration file.
        rad_file (str): Path to radiation output file.

    Returns:
        np.ndarray: Loaded flux data.
    """
    run_psg_docker(cfg_file, rad_file)
    data = np.loadtxt(rad_file, comments="#")
    return data


def find_average_fluxes(wave, flux, edges):
    """
    Find average fluxes within specified wavelength edges.

    Args:
        wave (np.ndarray): Wavelength array.
        flux (np.ndarray): Flux array.
        edges (list): List containing min and max wavelength edges.

    Returns:
        float: Average flux within the specified edges.
    """
    # find edges of bin over which to average result
    lam_min_idx = np.argmin(np.abs(wave - edges[0]))
    lam_max_idx = np.argmin(np.abs(wave - edges[1]))
    edges_idx = [lam_min_idx, lam_max_idx]
    # average flux over bandwidth
    return np.mean(flux[lam_min_idx:lam_max_idx])


def calculate_deltalambda_edges(lam, bandwidth, resolution):
    """
    Calculate wavelength edges based on bandwidth and resolution.

    Args:
        lam (float): Central wavelength.
        bandwidth (float): Bandwidth.
        resolution (float): Spectral resolution.

    Returns:
        tuple: Minimum wavelength, maximum wavelength, and delta lambda.
    """
    deltalambda = np.min(
        [
            lam / resolution,
            bandwidth * lam,
        ]
    )
    lam_min = lam - deltalambda / 2
    lam_max = lam + deltalambda / 2
    return lam_min, lam_max, deltalambda


def lambert_phase_function(alpha):
    """
    Calculate Lambert phase function.

    Args:
        alpha (float): Phase angle in radians.

    Returns:
        float: Lambert phase function value.
    """
    return (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / np.pi


def lambertphase_calculate_spectrum(
    geometric_albedo_spectrum,
    true_anomaly,
    inclination,
    rp,
    a,
    UpOmega,
    omega,
    d,
):
    """
    Calculate spectrum by multiplying geometric albedo and Lambert phase function.

    Args:
        geometric_albedo_spectrum (np.ndarray): Geometric albedo spectrum.
        true_anomaly (float): True anomaly in degrees.
        inclination (float): Inclination in degrees.
        rp (float): Planet radius in meters.
        a (float): Semi-major axis in AU.
        UpOmega (float): Longitude of ascending node in radians.
        omega (float): Argument of periapsis in radians.
        d (float): Observer distance in meters.

    Returns:
        np.ndarray: Scaled spectrum.
    """
    x, y, z = calculate_orbit_location(
        r=a * const.au,
        inclination=np.radians(inclination),
        true_anomaly=np.radians(true_anomaly),
        UpOmega=UpOmega,
        omega=omega,
    )

    # Calculate phase angle
    phase_angle = calculate_apparent_phase(x, y, z, d)

    # Calculate phase function
    phase_func = lambert_phase_function(np.radians(phase_angle))

    # Calculate (Rp/a)^2 factor
    r_in_m = (a * const.au).to(u.m).value
    rp_a_squared = (rp / r_in_m) ** 2

    # Scale the spectrum
    scaled_spectrum = geometric_albedo_spectrum * phase_func * rp_a_squared

    return scaled_spectrum


def psg_calculate_and_store_spectra(
    orbit_metadata, planet_template, base_output_dir, recalculate=False
):
    """
    Calculate and store spectra using PSG.

    Args:
        orbit_metadata (pd.DataFrame): Orbital metadata.
        planet_template (str): Path to planet template file.
        base_output_dir (str): Base output directory.
        recalculate (bool): Whether to recalculate existing spectra.

    Returns:
        pd.DataFrame: Spectrum metadata.
    """
    os.makedirs(base_output_dir, exist_ok=True)

    # Read the template cfg file
    with open(planet_template, "r") as file:
        template_cfg = [line for line in file]

    spectra = []

    for _, row in orbit_metadata.iterrows():
        a, i, t = row["a"], row["i"], row["t"]

        param_str = f"a{a:.2f}_i{i:.2f}_t{t:.2f}"
        template_cfg = mod_cfg_line(template_cfg, "<OBJECT-STAR-DISTANCE>", str(a))
        if i > 90:
            i = 180 - i
        template_cfg = mod_cfg_line(template_cfg, "<OBJECT-INCLINATION>", str((i)))
        template_cfg = mod_cfg_line(template_cfg, "<OBJECT-SEASON>", str((t)))

        cfg_file = os.path.join(base_output_dir, f"cfg_{param_str}.txt")
        rad_file = os.path.join(base_output_dir, f"rad_{param_str}.txt")
        if recalculate or not os.path.isfile(rad_file):
            print(f"Calculating spectrum for {param_str} and saving to {rad_file}")

            # Write the cfg file
            write_config(cfg_file, template_cfg)

            # Run PSG
            run_psg_docker(cfg_file, rad_file)

        spectra.append(
            {"index": row["index"], "a": a, "i": i, "t": t, "file": rad_file}
        )

    # Save spectrum metadata
    spectrum_metadata = pd.DataFrame(spectra)
    metadata_file = os.path.join(base_output_dir, "spectrum_metadata.csv")
    spectrum_metadata.to_csv(metadata_file, index=False)

    return spectrum_metadata


def lambertphase_calculate_and_store_spectra(
    orbit_metadata,
    planet_template,
    base_output_dir,
    planet_radius,
    observer_distance,
    wavelength_rebin,
    star_spectrum_rebin,
    recalculate=False,
):
    """
    Calculate and store spectra using Lambert phase function method.

    Args:
        orbit_metadata (pd.DataFrame): Orbital metadata.
        planet_template (str): Path to planet template file.
        base_output_dir (str): Base output directory.
        planet_radius (float): Planet radius in meters.
        observer_distance (float): Observer distance in meters.
        wavelength_rebin (np.ndarray): Wavelength array for rebinning.
        star_spectrum_rebin (np.ndarray): Rebinned star spectrum.
        recalculate (bool): Whether to recalculate existing spectra.

    Returns:
        pd.DataFrame: Spectrum metadata.
    """
    os.makedirs(base_output_dir, exist_ok=True)

    # Read geometric albedo spectrum from the template file
    geometric_albedo_spectrum = np.loadtxt(planet_template)

    spectra = []
    for _, row in orbit_metadata.iterrows():
        a, i, t = row["a"], row["i"], row["t"]
        param_str = f"a{a:.2f}_i{i:.2f}_t{t:.2f}"
        rad_file = os.path.join(base_output_dir, f"lambertmethod_{param_str}.txt")

        if recalculate or not os.path.isfile(rad_file):
            print(f"Calculating spectrum for {param_str} and saving to {rad_file}")

            # Calculate spectrum using lambertphase_calculate_spectrum
            spectrum = lambertphase_calculate_spectrum(
                geometric_albedo_spectrum[:, 1],
                true_anomaly=t,
                inclination=i,
                rp=planet_radius,
                a=a,
                UpOmega=0,
                omega=0,
                d=observer_distance,
            )
            # Rebin to actual wavelengths
            albedo_rebin = spectres.spectres(
                wavelength_rebin,
                geometric_albedo_spectrum[:, 0],
                geometric_albedo_spectrum[:, 1],
            )
            spectrum_rebin = spectres.spectres(
                wavelength_rebin,
                geometric_albedo_spectrum[:, 0],
                spectrum,
            )
            # Add the calculated spectrum as the third column of geometric_albedo_spectrum
            if "contrast" in base_output_dir:
                new_spectrum = np.column_stack(
                    (
                        wavelength_rebin,
                        albedo_rebin,
                        star_spectrum_rebin,
                        spectrum_rebin,
                    )
                )
            elif "Wm2um" in base_output_dir:
                new_spectrum = np.column_stack(
                    (
                        wavelength_rebin,
                        albedo_rebin,
                        star_spectrum_rebin,
                        spectrum_rebin * star_spectrum_rebin,
                    )
                )
            else:
                raise KeyError("Unknown location name.")
            # Remove rows with NaN values
            new_spectrum = new_spectrum[~np.isnan(new_spectrum).any(axis=1)]
            np.savetxt(rad_file, new_spectrum)
        spectra.append(
            {"index": row["index"], "a": a, "i": i, "t": t, "file": rad_file}
        )

    # Save spectrum metadata
    spectrum_metadata = pd.DataFrame(spectra)
    metadata_file = os.path.join(base_output_dir, "spectrum_metadata.csv")
    spectrum_metadata.to_csv(metadata_file, index=False)

    return spectrum_metadata
