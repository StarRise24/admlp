import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, griddata
from tqdm import tqdm
from scipy.stats import gaussian_kde
import seaborn as sns
from matplotlib.collections import LineCollection

PLOT_INDIVIDUAL = True

def get_nuscenes_data():
    data_dict = pickle.load(open('stp3_val/data_nuscene.pkl', 'rb'))
    return list(data_dict.values())

def get_nuscenes_occupancy():
    occ_dict = pickle.load(open('stp3_val/stp3_occupancy.pkl', 'rb'))
    
def get_carla_data():
    return pickle.load(open('data.pkl', 'rb'))
    #return pickle.load(open('data.pkl', 'rb'))


def get_data_points(dataset, key):
    if PLOT_INDIVIDUAL:
        if key in ['x', 'gt']:
            num_rows = 4 if key == 'x' else 7  # 4 rows for 'x' and 7 rows for 'gt'
            data_points = [[[] for _ in range(3)] for _ in range(num_rows)]  # 3 indices for each row
        else:
            data_points = [[] for _ in range(3)]  # 3 dimensions for 'v', 'a', 'cmd'

        # Accumulate data for each index of each row
        for entry in dataset:
            if key in ['x', 'gt']:
                for row_index, row in enumerate(entry[key]):
                    for i in range(3):
                        data_points[row_index][i].append(row[i])
            else:
                for i in range(3):
                    data_points[i].append(entry[key][i])

        return data_points


    else:
        data_points = [[] for _ in range(3)]  # Initialize 3 lists for 3 indices

        # Accumulate data for each index/dimension
        for entry in dataset:
            if key in ['x', 'gt']:
                for row in entry[key]:
                    for i in range(3):
                        data_points[i].append(row[i])
            else:  # For 'v', 'a', 'cmd'
                for i in range(3):
                    data_points[i].append(entry[key][i])
        return data_points

def plot_histograms(dataset1, dataset2, keys):
    for key in keys:
        data_points1 = get_data_points(dataset1, key)
        data_points2 = get_data_points(dataset2, key)

        if PLOT_INDIVIDUAL:
            if key in ['x', 'gt']:
                for row_index in range(len(data_points1)):
                    for i in range(3):
                        plt.figure(figsize=(10, 4))

                        plt.subplot(1, 2, 1)
                        plt.hist(data_points1[row_index][i], bins=20, alpha=0.7, label=f'Dataset 1 - {key} Row {row_index}[{i}]')
                        plt.legend()

                        plt.subplot(1, 2, 2)
                        plt.hist(data_points2[row_index][i], bins=20, alpha=0.7, color='orange', label=f'Dataset 2 - {key} Row {row_index}[{i}]')
                        plt.legend()

                        plt.suptitle(f'Histogram for {key} Row {row_index} Index {i}')
                        plt.show()
            else:
                for i in range(3):
                    plt.figure(figsize=(10, 4))

                    plt.subplot(1, 2, 1)
                    plt.hist(data_points1[i], bins=20, alpha=0.7, label=f'Dataset 1 - {key}[{i}]')
                    plt.legend()

                    plt.subplot(1, 2, 2)
                    plt.hist(data_points2[i], bins=20, alpha=0.7, color='orange', label=f'Dataset 2 - {key}[{i}]')
                    plt.legend()

                    plt.suptitle(f'Histogram for {key} Dimension {i}')
                    plt.show()

        else:
            num_plots = len(data_points1)
            for i in range(num_plots):
                plt.figure(figsize=(10, 4))

                plt.subplot(1, 2, 1)
                plt.hist(data_points1[i], bins=40, alpha=0.7, label=f'CARLA - {key}[{i}]')
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.hist(data_points2[i], bins=40, alpha=0.7, color='orange', label=f'NUSCENES - {key}[{i}]')
                plt.legend()

                plt.suptitle(f'Histogram for {key} Dimension {i}')
                plt.show()


def plot2dtraj(dataset):
    plt.figure(figsize=(10, 6))

    # Create a density map
    all_x = []
    all_y = []
    for traj in dataset:
        all_x.extend([point[0] for point in traj['gt']])
        all_y.extend([point[1] for point in traj['gt']])
    
    density, xedges, yedges = np.histogram2d(all_x, all_y, bins=[100, 100], density=True)
    
    # Normalize the density values to be between 0 and 1, with a base value to avoid very low opacities
    base_value = 0.05
    density_normalized = (density + base_value) / (np.max(density) + base_value)

    for i in tqdm(range(len(dataset))):
        x = np.array([point[0] for point in dataset[i]['gt']])
        y = np.array([point[1] for point in dataset[i]['gt']])
        if len(x) < 7:
            continue

        # Parametric variable
        t = np.linspace(0, 1, len(dataset[i]['gt']))

        # Create splines for x and y
        spline_x = CubicSpline(t, x)
        spline_y = CubicSpline(t, y)

        # Generate a smooth curve
        t_smooth = np.linspace(0, 1, 100)
        x_smooth = spline_x(t_smooth)
        y_smooth = spline_y(t_smooth)

        # Calculate the normalized average density for this trajectory
        hist_x = np.searchsorted(xedges, x_smooth, side="right") - 1
        hist_y = np.searchsorted(yedges, y_smooth, side="right") - 1

        # Ensure indices are within bounds
        hist_x = np.clip(hist_x, 0, density.shape[0] - 1)
        hist_y = np.clip(hist_y, 0, density.shape[1] - 1)

        traj_density = np.mean(density_normalized[hist_x, hist_y])

        plt.plot(x_smooth, y_smooth, alpha=traj_density, color='blue', linewidth=0.5)

    plt.xlabel('Relative X Coordinate')
    plt.ylabel('Distance')
    plt.title('Trajectories Plot')
    plt.show()



def plot_l2_errors():
    l2s = pickle.load(open('l2_errors.pkl', 'rb'))

    l2_errors_1s = l2s['1s']
    l2_errors_2s = l2s['2s']
    l2_errors_3s = l2s['3s']

    plt.figure(figsize=(8, 12))

    # Plot for L2 Errors for 1s
    plt.subplot(3, 1, 1)
    plt.hist(l2_errors_1s, bins=60, color='blue', edgecolor='black')
    plt.title('L2 Errors for 1s')
    plt.ylabel('Frequency')
    plt.xlabel('Error')

    # Plot for L2 Errors for 2s
    plt.subplot(3, 1, 2)
    plt.hist(l2_errors_2s, bins=60, color='green', edgecolor='black')
    plt.title('L2 Errors for 2s')
    plt.ylabel('Frequency')
    plt.xlabel('Error')

    # Plot for L2 Errors for 3s
    plt.subplot(3, 1, 3)
    plt.hist(l2_errors_3s, bins=60, color='red', edgecolor='black')
    plt.title('L2 Errors for 3s')
    plt.ylabel('Frequency')
    plt.xlabel('Error')

    plt.tight_layout()
    plt.show()


# Retrieve datasets
carla_dataset = get_carla_data()
nuscenes_dataset = get_nuscenes_data()
nuscenes_occ = get_nuscenes_occupancy()

# List of keys to process
keys = ['x', 'a', 'v', 'cmd', 'gt']
#keys =['v'] 

print("CARLA LENGTH: ", len(carla_dataset))
print("NUSCENE LENGTH: ", len(nuscenes_dataset))

#plot_histograms(carla_dataset, nuscenes_dataset, keys)
#plot2dtraj(carla_dataset)
plot_l2_errors()
#compare_occupancy(carla_dataset, )

