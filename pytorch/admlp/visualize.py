import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, griddata
from tqdm import tqdm
from scipy.stats import gaussian_kde
import seaborn as sns
from matplotlib.collections import LineCollection

PLOT_INDIVIDUAL = False

def get_nuscenes_data():
    data_dict = pickle.load(open('stp3_val/data_nuscene.pkl', 'rb'))
    data = list(data_dict.values())
    for i in range(len(data)):
        data[i]['gt'] = data[i]['gt'][1:]
    return data
    # return list(data_dict.values())

def get_nuscenes_occupancy():
    occ_dict = pickle.load(open('stp3_val/stp3_occupancy.pkl', 'rb'))
    
def get_carla_data():
    data = pickle.load(open('data_garage.pkl', 'rb'))
    # for i in range(len(data)):
    #     data[i]['gt'] = data[i]['gt'][1:]
    return data
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
                    value = entry[key][i]
                    data_points[i].append(value)

        return data_points


    else:
        data_points = [[] for _ in range(3)]  # Initialize 3 lists for 3 indices
        feature_counts = [0, 0, 0] if key == 'cmd' else [0, 0, 0, 0]

        # Accumulate data for each index/dimension

        for entry in dataset:
            if key in ['x', 'gt']:
                for row in entry[key]:
                    if key == 'x' and row[1] > 0.2:
                        continue
                    if key == 'gt' and row[0] < -0.1:
                        continue 
                    for i in range(3):
                        data_points[i].append(row[i])
            elif key in ['v', 'a']:
                for i in range(3):
                    value = entry[key][i]
                    if key == 'v' and i == 0 and value < 0:
                        #value = 0
                        continue
                    if key == 'a' and i == 0 and np.abs(value) > 5:
                        continue
                    if key == 'a' and i == 1 and np.abs(value) > 6:
                        continue
                    data_points[i].append(value)
            elif key in ['cmd', 'target_speed'] :
                vec = entry[key]
                for i, value in enumerate(vec):
                    feature_counts[i] += value

        if key in ['cmd', 'target_speed']:
            return feature_counts
        else:
            return data_points

def plot_histograms(dataset1, dataset2, keys):
    for key in keys:
        data_points1 = get_data_points(dataset1, key)
        data_points2 = get_data_points(dataset2, key)

        if PLOT_INDIVIDUAL:
            if key in ['x', 'gt']:
                for row_index in range(len(data_points1)):
                    for i in range(3):
                        plt.figure(figsize=(12, 5))

                        ax1 = plt.subplot(1, 2, 1)
                        ax1.hist(data_points1[row_index][i], bins=np.linspace(np.min(data_points1[row_index][i]), np.max(data_points1[row_index][i]), 21), color='skyblue')
                        ax1.legend(loc='upper right', bbox_to_anchor=(1.12, 1))
                        ax1.grid(False)
                        plt.xlabel('Value')
                        plt.ylabel('Frequency')

                        ax2 = plt.subplot(1, 2, 2)
                        ax2.hist(data_points2[row_index][i], bins=np.linspace(np.min(data_points2[row_index][i]), np.max(data_points2[row_index][i]), 21), color='lightgreen')
                        ax2.legend(loc='upper right', bbox_to_anchor=(1.12, 1))
                        ax2.grid(False)
                        plt.xlabel('Value')
                        plt.ylabel('Frequency')

                        plt.suptitle(f'Histogram for {key} Row {row_index} Index {i}')
                        plt.tight_layout()
                        plt.show()
            else:
                for i in range(3):
                    plt.figure(figsize=(12, 5))

                    ax1 = plt.subplot(1, 2, 1)
                    ax1.hist(data_points1[i], bins=40, color='skyblue')
                    ax1.legend(loc='upper right', bbox_to_anchor=(1.12, 1))
                    ax1.grid(False)
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')

                    ax2 = plt.subplot(1, 2, 2)
                    ax2.hist(data_points2[i], bins=40, color='lightgreen')
                    ax2.legend(loc='upper right', bbox_to_anchor=(1.12, 1))
                    ax2.grid(False)
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')

                    plt.suptitle(f'Histogram for {key} Dimension {i}')
                    plt.tight_layout()
                    plt.show()

        else:
            num_plots = len(data_points1)
            for i in range(num_plots):
                plt.figure(figsize=(12, 5))

                ax1 = plt.subplot(1, 2, 1)
                ax1.hist(data_points1[i], bins=40, color='cornflowerblue')
                ax1.legend(loc='upper right', bbox_to_anchor=(1.12, 1))
                ax1.grid(False)
                plt.xlabel('Value')
                plt.ylabel('Frequency')

                ax2 = plt.subplot(1, 2, 2)
                ax2.hist(data_points2[i], bins=40, color='orange')
                ax2.legend(loc='upper right', bbox_to_anchor=(1.12, 1))
                ax2.grid(False)
                plt.xlabel('Value')
                plt.ylabel('Frequency')

                plt.suptitle(f'Histogram for {key} Dimension {i}')
                plt.tight_layout()
                plt.show()


def plot_cmd(dataset):
    feature_counts = get_data_points(dataset=dataset, key='cmd')
    feature_labels = ['Left', 'Straight', 'Right']

    # Plotting the bar plot
    plt.bar(feature_labels, feature_counts, color='orange')
    plt.ylabel('Counts')
    plt.title('Distribution of High Level Commands')
    plt.show()

def plot2dtraj(dataset, entangled=True):
    dataset = dataset[:50000] 
    plt.figure(figsize=(8, 6))
    key = 'gt' if entangled else 'gt_path'

    # Create a density map
    all_x = []
    all_y = []
    for traj in dataset:
        y = [point[0] for point in traj[key]]
        # if (np.array(y) > 15).any():
        #     continue
        all_x.extend([point[1] for point in traj[key]])
        all_y.extend(y)
    
    density, xedges, yedges = np.histogram2d(all_x, all_y, bins=[100, 100], density=True)
    
    # Normalize the density values to be between 0 and 1, with a base value to avoid very low opacities
    base_value = 0.05
    density_normalized = (density + base_value) / (np.max(density) + base_value)

    for i in tqdm(range(len(dataset))):
        x = np.array([point[1] for point in dataset[i][key]])
        y = np.array([point[0] for point in dataset[i][key]])
        if len(x) < 6:
            continue
        # if (y > 15).any():
        #     continue

        # Parametric variable
        t = np.linspace(0, 1, len(dataset[i][key]))

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

    plt.xlabel('Distance')
    plt.ylabel('Distance')
    plt.title('Trajectories Plot')
    plt.show()



def plot_l2_errors():
    l2s = pickle.load(open('l2_errors.pkl', 'rb'))

    l2_errors_1s = l2s['1s']
    l2_errors_2s = l2s['2s']
    l2_errors_3s = l2s['3s']

    plt.figure(figsize=(5, 3))
    plt.hist(l2_errors_3s, bins=60, color='cornflowerblue')
    plt.title('L2 Errors for 3s')
    plt.ylabel('Frequency')
    plt.xlabel('Error')

    plt.tight_layout()
    plt.show()

def plot_gt_speed(dataset):
    feature_counts = get_data_points(dataset=dataset, key='target_speed')
    total_counts = sum(feature_counts)
    feature_labels = ['0m/s', '2m/s', '5m/s', '8m/s']
    
    # Convert counts to percentages
    feature_percentages = [(count / total_counts) * 100 for count in feature_counts]

    # Plotting the bar plot
    plt.bar(feature_labels, feature_percentages, color='cornflowerblue')
    plt.title('Distribution of Target Speeds')

    plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()])
    plt.show()


# Retrieve datasets
carla_dataset = get_carla_data()
nuscenes_dataset = get_nuscenes_data()
nuscenes_occ = get_nuscenes_occupancy()

# List of keys to process
# keys = ['x', 'a', 'v', 'cmd', 'gt']
keys =['gt'] 

print("CARLA LENGTH: ", len(carla_dataset))
print("NUSCENE LENGTH: ", len(nuscenes_dataset))

#plot_histograms(carla_dataset, nuscenes_dataset, keys)
#plot_cmd(nuscenes_dataset)
#plot_gt_speed(carla_dataset)

plot2dtraj(carla_dataset, entangled=False)
#plot_l2_errors()
#compare_occupancy(carla_dataset, )

