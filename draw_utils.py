import pandas as pd
from shapely import wkt
from shapely.geometry import Polygon, Point
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from datetime import datetime, timedelta
import seaborn as sns

color_palette = {
    "žuta": (212, 202, 20),
    "narančasta": (255, 120, 3),
    "crvena": (214, 21, 50),
    "zelena": (40, 200, 50),
    "svijetlo plava": (0, 255, 255),
    "plava": (70, 100, 170),
    "roza": (255, 140, 200),
    "ljubičasta": (100, 70, 212),
    "smeđa": (102, 70, 20),
    "bijela": (255, 255, 255),
    "crna": (0, 0, 0),
    "siva": (128, 128, 128),
    "bež": (212, 200, 170),
}

color_palette_english = {
    "yellow": (212, 202, 20),
    "orange": (255, 120, 3),
    "red": (214, 21, 50),
    "green": (40, 200, 50),
    "cyan": (0, 255, 255),
    "blue": (0, 0, 255),
    "pink": (255, 140, 200),
    "purple": (100, 70, 212),
    "brown": (102, 70, 20),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "grey": (128, 128, 128),
    "beige": (212, 200, 170),
}

def draw_flags(file: str, zone: str):
    zone_a = pd.read_csv(file)
    lat_lon = zone_a[['Lat', 'Lon']]
    lat_lon = lat_lon[(lat_lon['Lat'] >= 45.22) & (lat_lon['Lon'] >= 14.15)]
    plt.scatter(lat_lon['Lon'], lat_lon['Lat'], label=f'Zone {zone} markers', alpha=0.5)

def plot_points(file: str, zone: str):
    points_df = pd.read_csv(file)
    plt.scatter(points_df['Longitude'], points_df['Latitude'], alpha=0.5)

def get_zone(file):
    zone_a_wkt = pd.read_csv(file)
    polygon = wkt.loads(zone_a_wkt['WKT'][0])

    return polygon

def draw_zone(file: str, label: str):
    polygon = get_zone(file)
    x, y = polygon.exterior.xy

    plt.plot(x, y, label=label)

def color_zone(file: str, label: str, color: str):
    polygon = get_zone(file)
    x, y = polygon.exterior.xy
    plt.fill(x, y, alpha=0.3, color=color, label=label)

def plot_uav_trajectory(directory, name):
    """
    Plots the UAV trajectory from .npz files in the specified directory.
    
    Args:
        directory (str): The directory containing the .npz files.
    """
    
    # List all .npz files in the directory
    npz_files = [f for f in os.listdir(directory) if f.endswith('.npz')]
    latitudes = []
    longitudes = []

    # Read and store latitudes and longitudes from each .npz file
    for npz_file in npz_files:
        file_path = os.path.join(directory, npz_file)
        with np.load(file_path) as data:
            lat_lon = data["current_uav_states"][0][:2]
            if not np.isinf(lat_lon).any():
                latitudes.append(lat_lon[0])
                longitudes.append(lat_lon[1])

    # Plot the trajectory    
    plt.plot(longitudes, latitudes, linestyle='-', label=name)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'UAV Trajectory {name}')
    plt.ticklabel_format(useOffset=False)  # Disable scientific notation for x-axis

def is_point_in_polygon(point, polygon):
    return Polygon(polygon).contains(Point(point))

def get_exif_data(image):
    exif_data = {}
    info = image._getexif()
    if info:
        for tag, value in info.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == "GPSInfo":
                gps_data = {}
                for t in value:
                    sub_tag = GPSTAGS.get(t, t)
                    gps_data[sub_tag] = value[t]
                exif_data[tag_name] = gps_data
            else:
                exif_data[tag_name] = value
    return exif_data

def get_lat_lon(exif_data):
    lat = None
    lon = None
    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]
        gps_lat = gps_info.get("GPSLatitude")
        gps_lat_ref = gps_info.get("GPSLatitudeRef")
        gps_lon = gps_info.get("GPSLongitude")
        gps_lon_ref = gps_info.get("GPSLongitudeRef")

        if gps_lat and gps_lat_ref and gps_lon and gps_lon_ref:
            lat = convert_to_degrees(gps_lat)
            if gps_lat_ref != "N":
                lat = 0 - lat

            lon = convert_to_degrees(gps_lon)
            if gps_lon_ref != "E":
                lon = 0 - lon
    return lat, lon

def convert_to_degrees(value):
    d = float(value[0])
    m = float(value[1])
    s = float(value[2])
    return d + (m / 60.0) + (s / 3600.0)

def get_colors(SEARCH_EXPERIMENT: str, color_palette: dict):
    labels_path = f"./ucka2/{SEARCH_EXPERIMENT}_labels"
    files = os.listdir(labels_path)

    # Create a new dataframe to store RGB values
    hsl_df = pd.DataFrame(columns=['Label', 'color'])

    print(files, hsl_df)
    for file in files:
        with open(f'{labels_path}/{file}', 'r') as f:
            lines = f.readlines()
            image_file_name = file.replace('.txt', '')

            image_file = f'./ucka2/{SEARCH_EXPERIMENT}/{image_file_name}.JPG'
            image = cv2.imread(image_file)

            # Convert the image to RGB color space
            hsl_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for i, line in enumerate(lines):
                if line.startswith('0'):
                    # Extract the bounding box coordinates
                    _, x_center, y_center, width, height = map(float, line.split())
                    
                    # Convert to rectangle coordinates
                    img_height, img_width = image.shape[:2]
                    x_center *= img_width
                    y_center *= img_height
                    width *= img_width
                    height *= img_height
                    
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    
                    # Extract the region of interest
                    roi = hsl_image[y1:y2, x1:x2]

                    cv2.imwrite("test.png", roi)
                    
                    # Calculate the average RGB values
                    mean_color = np.mean(roi, axis=(0, 1))  # Average across height and width
                    median_color = np.median(roi, axis=(0, 1))  # Median across height and width

                    color1, color2, color3, color4, color5 = find_nearest_color(mean_color, color_palette)
                    color1, color2, color3, color4, color5 = find_nearest_color(median_color, color_palette)
                    
                    hsl_df = pd.concat([hsl_df, pd.DataFrame({'Label': [f'{image_file_name}_{i+1}'], 'color': [[color1, color2, color3, color4, color5]]})], ignore_index=True)

    return hsl_df

# Function to find the nearest color
def find_nearest_color(color, palette):
    color_values = np.array(list(palette.values()))
    color_names = list(palette.keys())
    distances = np.linalg.norm(color_values - color, axis=1)
    nearest_indices = np.argsort(distances)[:5]
    nearest_colors = [color_names[i] for i in nearest_indices]
    return nearest_colors

def sort_people(new_df):
    # Create a dictionary to store the top persons for each label
    top_persons_per_label = {}

    # Iterate through each label (column) in new_df_transposed
    for label in new_df.columns[1:]:  # Skip the 'index' column
        sorted_indices = new_df[label].argsort()[::-1]
        
        # Get the persons for the current label
        top_persons = new_df['Ime Prezime'].iloc[sorted_indices].values
        
        # Store the persons in the dictionary
        top_persons_per_label[label] = top_persons[:12]

    # Print the persons for each label
    for label, persons in top_persons_per_label.items():
        print(f"Label: {label}")
        print(f"Persons: {persons}")
        print()

    return top_persons_per_label

def transpose_matrix(new_df):
    new_df_copy = new_df.copy()
    new_df_copy.set_index('Ime Prezime', inplace=True)
    new_df_transposed = new_df_copy.transpose()
    new_df_transposed.reset_index(inplace=True)

    return new_df_transposed

# Haversine function to calculate distance between two lat/lon points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_distance(ref_lat, ref_lon, person_lat, person_lon):
    # Calculate the distance from the reference point to the detected person
    distance = haversine(ref_lat, ref_lon, person_lat, person_lon)
    
    return distance

def calculate_probability(ref_lat, ref_lon, person_lat, person_lon, sigma=300):
    # Calculate the distance from the reference point to the detected person
    distance = haversine(ref_lat, ref_lon, person_lat, person_lon)
    
    # Assume Gaussian distribution centered at the reference point
    probability = norm.pdf(distance, loc=0, scale=sigma)
    probability = probability / norm.pdf(0, loc=0, scale=sigma)
    
    return probability

def is_within_radius(point, image_point, radius=200):
    return calculate_distance(point[0], point[1], image_point[0], image_point[1]) < radius

def probability_is_within_radius(point, image_point, radius=200):
    if calculate_distance(point[0], point[1], image_point[0], image_point[1]) < radius:
        return calculate_probability(point[0], point[1], image_point[0], image_point[1])
    
def is_within_time_range(image_times, oznaka_time):
    oznaka_time = datetime.strptime(oznaka_time, '%H:%M')
    for image_time in image_times.split(', '):
        if image_time:
            image_time = datetime.strptime(image_time, '%H:%M:%S')
            if oznaka_time - timedelta(minutes=2) <= image_time <= oznaka_time + timedelta(minutes=2):
                return True
    return False

def get_labels(labels_path):
    labels = {}

    with open(labels_path, "r") as file:
        lines = file.readlines()
        i = 0
        for line in lines:
            line = line.strip().split()
            if len(line) == 5:
                id = line[0]
                x1 = float(line[1])
                y1 = float(line[2])
                width1 = float(line[3])
                height1 = float(line[4])
                labels[i] = {"id": id, "x": x1, "y": y1, "width": width1, "height": height1}
                i = i + 1

    return labels

def convert_to_yolov8_format(bbox, img_width, img_height):
    """
    Convert bounding box from OpenCV format to YOLOv8 format.
    
    Parameters:
    bbox (tuple): Bounding box in OpenCV format (x1, y1, x2, y2)
    img_width (int): Width of the image
    img_height (int): Height of the image
    
    Returns:
    tuple: Bounding box in YOLOv8 format (x_center, y_center, width, height)
    """
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    x_center = x1 + bbox_width / 2
    y_center = y1 + bbox_height / 2
    
    # Normalize the coordinates
    x_center /= img_width
    y_center /= img_height
    bbox_width /= img_width
    bbox_height /= img_height
    
    return np.round([x_center, y_center, bbox_width, bbox_height], 7)

def get_opencv_coords(data, width, height):
    x = data["x"]
    y = data["y"]
    w = data["width"]
    h = data["height"]

    centerx = x * width
    centery = y * height
    newx1 = centerx - w * width / 2
    newy1 = centery - h * height / 2
    newx2 = centerx + w * width / 2
    newy2 = centery + h * height / 2

    return newx1, newy1, newx2, newy2

def plot_gsd(df, save_image):
    groups = df['GSD'].tolist()

    print(df['GSD'].min())
    print(df['GSD'].max())

    plt.figure(figsize=(10, 5))
    plt.hist(groups, bins=np.arange(0, 7, 0.5), color='skyblue', edgecolor='black')
    plt.xlabel('GSD Groups')
    plt.ylabel('Number of images')
    plt.title('Number of images in each GSD group')
    plt.savefig(f'./{save_image}.png')
    plt.xlim(0, 6.5)
    plt.show()

def clean_m30_images(folder_names):
    for folder in folder_names:
        image_files = os.listdir(folder)

        for image_file in image_files:
            if image_file.endswith('T.JPG') or image_file.endswith('Z.JPG'):
                file_path = os.path.join(folder, image_file)
                os.remove(file_path)

def create_empty_labels(image_folder, label_folder):
    os.makedirs(label_folder, exist_ok=True)

    image_files = os.listdir(image_folder)

    for image_file in image_files:
        if image_file.endswith('.jpg'):
            label_file = image_file.replace('.jpg', '.txt')
            label_path = os.path.join(label_folder, label_file)
            
            if not os.path.exists(label_path):
                with open(label_path, 'w') as f:
                    pass

def plot_heatmap(df, label):
    plt.figure(figsize=(24, 14))
    sns.heatmap(df.set_index('Ime Prezime').astype(float), cmap='crest_r', annot=False, vmin=0, vmax=1)
    plt.ylabel('Person Index')
    plt.title(label)
    plt.show()

def plot_heatmap_transposed(df, label):
    plt.figure(figsize=(24, 14))
    sns.heatmap(df.set_index('index').astype(float), cmap='crest_r', annot=False, xticklabels=[i for i in range(1, len(df.columns))], vmin=0, vmax=1)
    plt.xlabel('Person Index')
    plt.ylabel('Label')
    plt.title(label)
    plt.show()

def get_labels_list(labels_path):
    # Filter files that contain rows beginning with 0 - images with labels
    filtered_files = []

    files = os.listdir(labels_path)
    for file in files:
        with open(f'{labels_path}/{file}', 'r') as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                if line.startswith('0'):
                    image_file_name = file.replace('.txt', '')
                    filtered_files.append(f'{image_file_name}_{i+1}')

    return filtered_files

def json_bbox_to_yolo_bbox(bbox, img_width, img_height):
    """
    Convert COCO JSON bbox format [x_min, y_min, width, height] to YOLO format [x_center, y_center, w, h] (normalized).
    bbox: list or tuple of [x_min, y_min, width, height]
    img_width: int, image width in pixels
    img_height: int, image height in pixels
    Returns: [x_center_norm, y_center_norm, w_norm, h_norm]
    """
    x_min, y_min, w, h = bbox
    x_center = x_min + w / 2
    y_center = y_min + h / 2
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return [x_center_norm, y_center_norm, w_norm, h_norm]

def yolo_bbox_to_opencv_bbox(yolo_bbox, img_width, img_height):
    """
    Convert YOLO bbox format [x_center, y_center, w, h] (normalized) to OpenCV format [x_min, y_min, x_max, y_max] (pixels).
    yolo_bbox: list or tuple of [x_center, y_center, w, h] (all normalized)
    img_width: int, image width in pixels
    img_height: int, image height in pixels
    Returns: [x_min, y_min, x_max, y_max] (all int)
    """
    x_center, y_center, w, h = yolo_bbox
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    w_px = w * img_width
    h_px = h * img_height

    x_min = int(round(x_center_px - w_px / 2))
    y_min = int(round(y_center_px - h_px / 2))
    x_max = int(round(x_center_px + w_px / 2))
    y_max = int(round(y_center_px + h_px / 2))

    return [x_min, y_min, x_max, y_max]
