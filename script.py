from marker_advection.advect_marker_locations import advect_marker_locations
import time
from datetime import datetime
import numpy as np
import os
from geo_converter import latlon_to_local, local_to_latlon

data_dir = f'D:/cres/marker_advection/data_dir' # measurements in npz format
case_dir = f'D:/cres/marker_advection/template_dir/case_Cres' # OpenFOAM case
advection_results_dir = f'D:/cres/marker_advection/advection_results_dir' # advected marker locations
advection_time_step = 600 # desired advection time step

src_dir = 'C:/Users/Stella/AOSeR Dropbox/Stella Dumencic/Cres_experiment/work_folder/search/flow_fields'
dst_dir = 'D:/cres/marker_advection/template_dir/case_Cres/0'
files = sorted(os.listdir(src_dir))

# Save targets to txt files
save_path_karlo = 'C:/Users/Stella/AOSeR Dropbox/Stella Dumencic/Cres_experiment/work_folder/monitoring/target_detection'
save_path_stella = './detected_targets'
os.makedirs(save_path_stella, exist_ok=True)

index = 9001
previous_timestamp = None

while True:

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nearest_file = None
    drifter_dt = datetime.strptime(str(timestamp), "%Y-%m-%d_%H-%M-%S")
    min_time_diff = float('inf')
    
    for f in files:
        try:
            # Extract timestamp from filename (assumes format 'U_YYYY-MM-DD_HH-MM-SS')
            file_ts = f.split('_', 1)[-1]
            file_dt = datetime.strptime(file_ts, "%Y-%m-%d_%H-%M-%S")
            if file_dt <= drifter_dt:
                diff = (drifter_dt - file_dt).total_seconds()
                if diff < min_time_diff:
                    min_time_diff = diff
                    nearest_file = f
        except Exception:
            continue

    print(f"Nearest file: {nearest_file} with time difference: {min_time_diff} seconds")

    # if nearest_file:
    #     shutil.copy(os.path.join(src_dir, nearest_file), os.path.join(dst_dir, 'U'))
    
    advect_marker_locations(case_dir, advection_time_step, data_dir, advection_results_dir, index+1, timestamp)
    index = index + 1

    base_point_test = [44.952006, 14.364774]

    npz_files = [f for f in os.listdir(advection_results_dir) if f.endswith('.npz')]
    npz_files.sort()
    latest_npz = npz_files[-1]
    latest_npz_data = np.load(os.path.join(advection_results_dir, latest_npz))
    latlon_points = latest_npz_data["drifter_location"]
    print(latlon_points)
    
    name = os.path.splitext(latest_npz)[0]
    
    now = timestamp
    drifter_location = local_to_latlon(latlon_points, base_point_test)
    np.savez(f'{name}_global.npz', drifter_location=drifter_location)    

    for index, point in enumerate(drifter_location):
        filename_karlo = os.path.join(save_path_karlo, f"Detection_{index+1:02d}_{now}.txt")
        with open(filename_karlo, 'w') as f:
            f.write(f"{point[0]} {point[1]}\n")

        filename_stella = os.path.join(save_path_stella, f"Detection_{index+1:02d}_{now}.txt")
        with open(filename_stella, 'w') as f_stella:
            f_stella.write(f"{point[0]} {point[1]}\n")

    if previous_timestamp is not None:
        for f in os.listdir(save_path_karlo):
            if f.endswith(f"{previous_timestamp}.txt"):
                os.remove(os.path.join(save_path_karlo, f))

    previous_timestamp = timestamp
        
    time.sleep(20)