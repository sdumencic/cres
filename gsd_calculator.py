import numpy as np

def calculate_gsd(fov_d, resolution, height):
    """
    :param fov_d: Diagonal FOV in degrees
    :param resolution: Resolution in pixels - numpy 1D array [number of horizontal pixels, number of vertical pixels]
    :param height: image capture height in meters
    :return: GSD in cm/px
    """

    fov_d_rad = np.deg2rad(fov_d)

    # Calculating horizontal and vertical FOV
    fov_h = 2 * np.arctan((np.tan(fov_d_rad/2)**2/(1 + (resolution[1]/resolution[0])**2))**0.5)
    fov_v = 2 * np.arctan((np.tan(fov_d_rad/2)**2/(1 + (resolution[0]/resolution[1])**2))**0.5)

    print(f"FOV H: {np.rad2deg(fov_h)}")
    print(f"FOV V: {np.rad2deg(fov_v)}")


    # Calculating gsd in horizontal image direction
    h_length = 2 * height * np.tan(fov_h/2)     # meters
    gsd_h = h_length / resolution[0] * 100  # gsd in cm/px

    print(f"H length: {h_length}")

    # Calculating gsd in horizontal image direction
    v_length = 2 * height * np.tan(fov_v / 2)
    gsd_v = v_length / resolution[1] * 100

    print(f"V length: {v_length}")

    # Since horizontal and vertical FOV are calculated from aspect ratio and diagonal fov -> gsd_vertical = gsd_horizontal
    return gsd_h


def calculate_gsd_for_M30_wide_camera(height):
    return calculate_gsd(84, np.array([4000, 3000]), height)

def calculate_gsd_for_M210_x5s_camera(height):
    return calculate_gsd(72, np.array([5280, 2970]), height)



if __name__ == "__main__":

    h = 85 # meters

    # print(f"M30T wide camera GSD at {h} meters: {calculate_gsd_for_M30_wide_camera(h)}")
    print(f"M210 X5S camera GSD at {h} meters: {calculate_gsd_for_M210_x5s_camera(h)}")

    # print(calculate_gsd(63.7, np.array([1920,1080]), h))









