import numpy as np
import pyproj


def get_utm_zone(longitude):
    # Calculate UTM zone based on the provided longitude
    zone_number = int((longitude + 180) / 6) + 1
    if zone_number > 60:
        zone_number = 60
    elif zone_number < 1:
        zone_number = 1
    return zone_number


def latlon_to_local(global_points, base_point, convert_point_list_to_point=True):
    """
    Converts a 2-D array (n,2) of n global points to local coordinate system centered around a base point [lat, lon].
    :param global_points: 2-D Array (n, 2) of n points [lat, lon] or an 1-D array representing a single point [lat, lon]
    :param base_point: Base point coordinates [lat, lon]
    :param convert_point_list_to_point: (Boolean) If global_points contains one point, and if True it converts list of
    1 point to single point: [[x, y]] -> [x, y]
    :return: If given an array of n global points it returns an array of n local points [x, y],
    but if given a single point it returns a single local point [x, y]
    """

    # Packing point into an array in case there is only one point
    if np.size(np.shape(global_points)) == 1:
        global_points = np.array([global_points])

    # Unpacking values
    lats = global_points[:, 0]
    lons = global_points[:, 1]

    # Calculate UTM zone for the base location
    base_utm_zone = get_utm_zone(base_point[1])  # Use the base longitude

    # Define the projection for the UTM zone
    utm_proj = pyproj.Proj(proj='utm', zone=base_utm_zone, ellps='WGS84')

    # Convert lat/lon to UTM coordinates
    utm_x, utm_y = utm_proj(lons, lats)

    # Offset UTM coordinates by the base point to get local coordinates
    local_x = utm_x - utm_proj(base_point[1], base_point[0])[0]
    local_y = utm_y - utm_proj(base_point[1], base_point[0])[1]

    converted_points = np.array([local_x, local_y]).T

    if convert_point_list_to_point:
        if np.shape(converted_points)[0] == 1:
            return converted_points[0]

    return converted_points


def local_to_latlon(local_points, base_point, convert_point_list_to_point=True):
    """
    Converts a 2-D array (n,2) of n local points to global (lat, lon)
    coordinate system centered around a base point [lat, lon].
    :param local_points: 2-D Array (n, 2) of n points [x, y] or an 1-D array representing a single point [x, y]
    :param base_point: Base point coordinates [lat, lon]
    :param convert_point_list_to_point: (Boolean) If global_points contains one point, and if True it converts list of
    1 point to single point: [[x, y]] -> [x, y]
    :return: If given an array of n local points it returns an array of n global points [lat, lon],
    but if given a single point it returns a single global point [lat, lon]
    """
    # Packing point into an array in case there is only one point
    if np.size(np.shape(local_points)) == 1:
        local_points = np.array([local_points])

    # Unpacking values
    local_x = local_points[:, 0]
    local_y = local_points[:, 1]

    # Calculate UTM zone for the base location
    base_utm_zone = get_utm_zone(base_point[1])  # Use the base longitude

    # Define the projection for the UTM zone
    utm_proj = pyproj.Proj(proj='utm', zone=base_utm_zone, ellps='WGS84')

    # Offset local coordinates to get UTM coordinates
    utm_x = local_x + utm_proj(base_point[1], base_point[0])[0]
    utm_y = local_y + utm_proj(base_point[1], base_point[0])[1]

    # Convert UTM coordinates to lat/lon
    lon, lat = utm_proj(utm_x, utm_y, inverse=True)

    converted_points = np.array([lat, lon]).T

    if convert_point_list_to_point:
        if np.shape(converted_points)[0] == 1:
            return converted_points[0]

    return converted_points


if __name__ == "__main__":
    base_point_test = [45.26398, 14.18947]

    # Convert lat/lon to local coordinates
    lat_test = 45.28182
    lon_test = 14.18918

    points = np.array([[45.1, 14.4],
                       [45.2, 14.4],
                       [45.4, 14.4],
                       [45.0, 14.1]])

    point = np.array([[44.952006, 14.364774]])

    # Global to local
    test_local_points = latlon_to_local(points, base_point_test)
    test_local_point = latlon_to_local(point, base_point_test)
    test_global_points = local_to_latlon(test_local_points, base_point_test)
    test_global_point = local_to_latlon(test_local_point, base_point_test)

    print("\n" + "*" * 50 + "\n" + "Initial global values\n" + "*" * 50 + "\n")
    print(f"Points = {points} \n\nPoint = {point}\n")
    print("*" * 50 + "\n" + "Global to local test\n" + "*" * 50 + "\n")
    print(f"Local points = {test_local_points} \n\nLocal point = {test_local_point}\n")
    print("*" * 50 + "\n" + "Back to global\n" + "*" * 50 + "\n")
    print(f"Global points = {test_global_points} \n\nGlobal point = {test_global_point}")


