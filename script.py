from marker_advection.advect_marker_locations import advect_marker_locations
import keyboard
import time

data_dir = f'D:/cres/marker_advection/data_dir' # measurements in npz format
case_dir = f'D:/cres/marker_advection/template_dir/case_Cres' # OpenFOAM case
advection_results_dir = f'D:/cres/marker_advection/advection_results_dir' # advected marker locations
advection_time_step = 600 # desired advection time step

gaussian_noise=False

name = 0
print("Press 'Esc' to exit the loop.")

while not keyboard.is_pressed('esc'):    

    advect_marker_locations(case_dir, advection_time_step, data_dir, advection_results_dir, name)
    name = name + 1
    time.sleep(10)