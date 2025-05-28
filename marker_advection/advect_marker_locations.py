import numpy as np
import pathlib
import pyvista as pv
import os


def advect_marker_locations(case_dir, advection_time_step, data_dir, advection_results_dir, name):

    print(f'Getting results from ({case_dir})')
    pathlib.Path(f'{case_dir}/case.foam').touch()
    reader = pv.OpenFOAMReader(f'{case_dir}/case.foam')
    reader.set_active_time_value(reader.time_values[-1])
    mesh = reader.read()
    internal_mesh = mesh["internalMesh"]

    # load the latest drifter location files
    latest_drifter_location_file = sorted(
        [entry.path for entry in os.scandir(data_dir) if
        entry.is_file() and entry.name.startswith('drifter_measurements') and entry.name.endswith('.npz')])[-1]

    # drifter measurements
    drifter_measurements = np.load(latest_drifter_location_file)
    drifter_location = drifter_measurements['drifter_location']
    location_array = np.array(drifter_location)

    advected_locations = []  # to store final positions

    sub_step = 1 # in order to check velocity field every 5 seconds
    n_sub_steps = advection_time_step // sub_step
    for i, initial_position in enumerate(location_array):
        # start with the initial position
        position = np.array([initial_position[0], initial_position[1], 0])
        for sub in range(n_sub_steps):
            # advect the particle for the defined time step
            location = pv.PolyData(position)
            samples = location.sample(internal_mesh)
            position += sub_step * samples['U'][0, :]

        advected_locations.append(position[:2])

    drifter_location = np.array(advected_locations)

    # check if the locations are inside CFD domain
    check_locations = np.hstack([drifter_location, np.zeros((drifter_location.shape[0], 1))])
    check_pd = pv.PolyData(check_locations)
    sampled = check_pd.sample(internal_mesh)

    # Create mask based on non-NaN velocity values
    mask = ~np.isnan(sampled['U'][:, 0])
    drifter_location = drifter_location[mask]

    np.savez_compressed(f'{advection_results_dir}/advected_marker_locations_{name}.npz', drifter_location=drifter_location)


if __name__ == '__main__':

    data_dir = f'C:/Users/stella/Documents/cres/marker_advection/data_dir' # measurements in npz format
    case_dir = f'C:/Users/stella/Documents/cres/marker_advection/template_dir/case_Cres' # OpenFOAM case
    advection_results_dir = f'C:/Users/stella/Documents/cres/marker_advection/advection_results_dir' # advected marker locations
    advection_time_step = 600 # desired advection time step

    gaussian_noise=False
    advect_marker_locations(case_dir, advection_time_step, data_dir, advection_results_dir, 1)



