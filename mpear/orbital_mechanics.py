import numpy as np
import astropy.constants as const
from astropy import units as u
import os
import pandas as pd

# ORBITAL CALCULATIONS


def to_arcsec(distance, observer_distance):
    """
    Convert a distance to arcseconds.

    Args:
        distance (astropy.units.Quantity): The distance to convert.
        observer_distance (astropy.units.Quantity): The distance of the observer.

    Returns:
        float: The angle in arcseconds.
    """
    return np.arctan(distance / observer_distance).to(u.arcsec).value


def calculate_orbit_location(r, inclination, true_anomaly, UpOmega, omega):
    """
    Calculate the position of an object in its orbit.

    Args:
        r (astropy.units.Quantity): The radius of the orbit.
        inclination (float): The inclination of the orbit in radians.
        true_anomaly (float): The true anomaly in radians.
        UpOmega (float): The longitude of the ascending node in radians.
        omega (float): The argument of periapsis in radians.

    Returns:
        tuple: The x, y, z coordinates of the object.
    """
    i_obs = np.pi - inclination  # Adjust inclination for negative Y axis
    x = r * (
        np.cos(UpOmega) * np.cos(omega + true_anomaly)
        - np.sin(UpOmega) * np.sin(omega + true_anomaly) * np.cos(i_obs)
    )
    y = r * (
        np.sin(UpOmega) * np.cos(omega + true_anomaly)
        + np.cos(UpOmega) * np.sin(omega + true_anomaly) * np.cos(i_obs)
    )
    z = r * np.sin(omega + true_anomaly) * np.sin(inclination)
    return x, y, z


def calculate_full_orbit(r, inclination, UpOmega, omega, d, save=False, filename=None):
    """
    Calculate the full orbit of an object.

    Args:
        r (astropy.units.Quantity): The radius of the orbit.
        inclination (float): The inclination of the orbit in radians.
        UpOmega (float): The longitude of the ascending node in radians.
        omega (float): The argument of periapsis in radians.
        d (astropy.units.Quantity): The distance to the observer.
        save (bool): Whether to save the orbit data to a file.
        filename (str): The filename to save the orbit data.

    Returns:
        numpy.ndarray: The orbit data in arcseconds.
    """
    orbit_data = []
    for true_anomaly in np.linspace(0, np.radians(360), 100):
        x, y, z = calculate_orbit_location(r, inclination, true_anomaly, UpOmega, omega)
        orbit_data.append([x.value, y.value, z.value])
    orbit_data = np.array(orbit_data)
    orbit_data = to_arcsec(orbit_data * u.m, d)
    if save:
        with open(filename, "w") as file:
            for point in orbit_data:
                file.write(f"{point[0]:.6f}\t{point[1]:.6f}\t{point[2]:.6f}\n")
    return orbit_data


def find_matching_orbits(
    target_position, d, a_range, i_range, t_range, UpOmega, omega, tolerance
):
    """
    Find orbits that match the target position within a given tolerance.

    Args:
        target_position (tuple): The target position in arcseconds.
        d (astropy.units.Quantity): The distance to the observer.
        a_range (numpy.ndarray): Range of semi-major axis values.
        i_range (numpy.ndarray): Range of inclination values.
        t_range (numpy.ndarray): Range of true anomaly values.
        UpOmega (float): The longitude of the ascending node in radians.
        omega (float): The argument of periapsis in radians.
        tolerance (float): The maximum allowed distance from the target position.

    Returns:
        tuple: Two 3D matrices containing distances and filtered distances.
        (max_matrix contains values only where tolerance condition is satisfied, useful
        for plotting. matrix contains all values instead.)
    """
    target_x, target_y = target_position
    matrix = np.zeros((len(i_range), len(t_range), len(a_range)))
    max_matrix = np.zeros((len(i_range), len(t_range), len(a_range)))

    for a_index, a in enumerate(a_range):
        for i_index, i in enumerate(i_range):
            for t_index, p in enumerate(t_range):
                r = a * const.au
                inclination = np.radians(i)
                true_anomaly = np.radians(p)
                x, y, z = calculate_orbit_location(
                    r, inclination, true_anomaly, UpOmega, omega
                )
                x_arcsec = to_arcsec(x, d)
                y_arcsec = to_arcsec(y, d)
                distance = np.sqrt(
                    (x_arcsec - target_x) ** 2 + (y_arcsec - target_y) ** 2
                )
                matrix[i_index, t_index, a_index] = distance
                max_matrix[i_index, t_index, a_index] = (
                    distance if distance <= tolerance else np.inf
                )

    return matrix, max_matrix


def find_matching_orbits_4d(
    target_position, d, a_range, i_range, t_range, UpOmega_range, omega, tolerance
):
    """
    Find orbits that match the target position within a given tolerance, including UpOmega as a variable.

    Args:
        target_position (tuple): The target position in arcseconds.
        d (astropy.units.Quantity): The distance to the observer.
        a_range (numpy.ndarray): Range of semi-major axis values.
        i_range (numpy.ndarray): Range of inclination values.
        t_range (numpy.ndarray): Range of true anomaly values.
        UpOmega_range (numpy.ndarray): Range of UpOmega values.
        omega (float): The argument of periapsis in radians.
        tolerance (float): The maximum allowed distance from the target position.

    Returns:
        tuple: Two 4D matrices containing distances and filtered distances.
    """
    # Unpack the target position tuple
    target_x, target_y = target_position
    matrix = np.zeros((len(i_range), len(t_range), len(a_range), len(UpOmega_range)))
    max_matrix = np.zeros(
        (len(i_range), len(t_range), len(a_range), len(UpOmega_range))
    )
    for om_index, om in enumerate(UpOmega_range):
        for a_index, a in enumerate(a_range):
            for i_index, i in enumerate(i_range):
                for t_index, t in enumerate(t_range):
                    r = a * const.au
                    inclination = np.radians(i)
                    true_anomaly = np.radians(t)
                    UpOmega = np.radians(om)
                    x, y, z = calculate_orbit_location(
                        r, inclination, true_anomaly, UpOmega, omega
                    )
                    x_arcsec = to_arcsec(x, d)
                    y_arcsec = to_arcsec(y, d)
                    # Calculate the distance from the target position
                    distance = np.sqrt(
                        (x_arcsec - target_x) ** 2 + (y_arcsec - target_y) ** 2
                    )
                    matrix[i_index, t_index, a_index, om_index] = distance
                    max_matrix[i_index, t_index, a_index, om_index] = (
                        distance if distance <= tolerance else np.inf
                    )
    return matrix, max_matrix


def load_or_calculate_matrix(
    filename,
    calculation_function,
    projected_distance,
    d,
    a_range,
    i_range,
    t_range,
    UpOmega_fixed_or_range,
    omega,
    tolerance,
):
    """
    Load existing matrix data from a file or calculate it if the file doesn't exist or axes don't match.

    Args:
        filename (str): The filename to load from or save to.
        calculation_function (function): The function to use for calculation if needed.
        projected_distance (tuple): The projected distance of the target.
        d (astropy.units.Quantity): The distance to the observer.
        a_range (numpy.ndarray): Range of semi-major axis values.
        i_range (numpy.ndarray): Range of inclination values.
        t_range (numpy.ndarray): Range of true anomaly values.
        UpOmega_fixed_or_range (float or numpy.ndarray): Fixed value or range of UpOmega values.
        omega (float): The argument of periapsis in radians.
        tolerance (float): The maximum allowed distance from the target position.

    Returns:
        tuple: Two matrices containing distances and filtered distances.
    """
    file_exists = os.path.isfile(filename)
    axes_match = False

    if file_exists:
        data = np.load(filename)
        matrix = data["matrix"]
        max_matrix = data["max_matrix"]
        index_x = data["index_x"]
        index_y = data["index_y"]
        index_z = data["index_z"]

        is_4d = "index_w" in data.files
        if is_4d:
            index_w = data["index_w"]

        # Check if axes match
        if (
            len(index_x) == len(i_range)
            and len(index_y) == len(t_range)
            and len(index_z) == len(a_range)
        ):
            axes_match = (
                np.allclose(index_x, i_range)
                and np.allclose(index_y, t_range)
                and np.allclose(index_z, a_range)
            )
            if is_4d:
                axes_match = (
                    axes_match
                    and len(index_w) == len(UpOmega_fixed_or_range)
                    and np.allclose(index_w, UpOmega_fixed_or_range)
                )
        data.close()

    if not file_exists or not axes_match:
        print(
            f"File {'exists but axes do not match' if file_exists else 'does not exist'}. Recalculating..."
        )
        matrix, max_matrix = calculation_function(
            projected_distance,
            d,
            a_range,
            i_range,
            t_range,
            UpOmega_fixed_or_range,
            omega,
            tolerance,
        )
        save_dict = {
            "matrix": matrix,
            "max_matrix": max_matrix,
            "index_x": i_range,
            "index_y": t_range,
            "index_z": a_range,
        }

        # Add fourth dimension if it's a 4D matrix
        if len(matrix.shape) == 4:
            save_dict["index_w"] = UpOmega_fixed_or_range
        np.savez(filename, **save_dict)
    else:
        print(f"Loaded existing data from {filename}")

    return matrix, max_matrix


def calculate_apparent_phase(x, y, z, distance):
    """
    Calculate the apparent phase angle of a planet.

    Args:
        x (astropy.units.Quantity): The x-coordinate of the planet.
        y (astropy.units.Quantity): The y-coordinate of the planet.
        z (astropy.units.Quantity): The z-coordinate of the planet.
        distance (astropy.units.Quantity): The distance to the observer.

    Returns:
        float: The apparent phase angle in degrees.
    """
    star_to_planet = np.array([x.value, y.value, z.value])
    planet_to_star = -star_to_planet
    star_to_observer = np.array([0, 0, distance.value])
    planet_to_observer = star_to_observer - star_to_planet
    cos_phase = np.dot(planet_to_observer, planet_to_star) / (
        np.linalg.norm(planet_to_star) * np.linalg.norm(planet_to_observer)
    )
    phase_angle = np.arccos(cos_phase)
    return np.degrees(phase_angle)


def calculate_and_store_orbits(
    matrix,
    tolerance,
    a_range,
    i_range,
    t_range,
    UpOmega,
    omega,
    d,
    base_output_dir,
    keyword="",
    recalculate=False,
):
    """
    Calculate and store orbits that match the given tolerance.

    Args:
        matrix (numpy.ndarray): The matrix of distances.
        tolerance (float): The maximum allowed distance from the target position.
        a_range (numpy.ndarray): Range of semi-major axis values.
        i_range (numpy.ndarray): Range of inclination values.
        t_range (numpy.ndarray): Range of true anomaly values.
        UpOmega (float): The longitude of the ascending node in radians.
        omega (float): The argument of periapsis in radians.
        d (astropy.units.Quantity): The distance to the observer.
        base_output_dir (str): The base directory to store orbit files.
        keyword (str): A keyword to add to the metadata filename.
        recalculate (bool): Whether to recalculate existing orbits.

    Returns:
        pandas.DataFrame: A DataFrame containing metadata for the calculated orbits.
    """
    os.makedirs(base_output_dir, exist_ok=True)
    selected_i, selected_t, selected_a = np.where(matrix < tolerance)
    orbits = []

    for index in range(len(selected_i)):
        a = a_range[selected_a[index]]
        i = i_range[selected_i[index]]
        t = t_range[selected_t[index]]
        param_str = f"a{a:.2f}_i{i:.2f}_t{t:.2f}"
        orbit_file = os.path.join(base_output_dir, f"orbit_{param_str}.txt")

        if not os.path.isfile(orbit_file) or recalculate:
            r = a * const.au
            inclination = np.radians(i)
            true_anomaly = np.radians(t)
            orbit = calculate_full_orbit(r, inclination, UpOmega, omega, d)
            np.savetxt(orbit_file, orbit, fmt="%.6f")
        else:
            orbit = np.loadtxt(orbit_file)
        orbits.append({"index": index, "a": a, "i": i, "t": t, "file": orbit_file})

    orbit_metadata = pd.DataFrame(orbits)
    metadata_file = base_output_dir + "/" + keyword + "orbit_metadata.csv"
    orbit_metadata.to_csv(metadata_file, index=False)

    return orbit_metadata
