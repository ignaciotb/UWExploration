import numpy as np

import matplotlib.pyplot as plt

from scipy.spatial import KDTree

from scipy.spatial.distance import cdist

from argparse import ArgumentParser



def _construct_query_tree(min_x: float, min_y: float, num_rows: float,

                          num_cols: float, resolution: float):

    x = np.linspace(min_x + (0.5 * resolution),

                    min_x + (num_rows - 1 + 0.5) * resolution, num_rows)

    y = np.linspace(min_y + (0.5 * resolution),

                    min_y + (num_cols - 1 + 0.5) * resolution, num_cols)

    xv, yv = np.meshgrid(x, y)

    queries = np.stack((xv.flatten(), yv.flatten()), axis=-1)

    queries_tree = KDTree(queries)

    return queries, queries_tree



def plot_consistency_matrix(consistency_matrix: np.ndarray):

    # Plot the consistency matrix, assumes that 0 values are no hits

    consistency_matrix[consistency_matrix == 0] = np.nan

    plt.contourf(consistency_matrix, cmap='jet', levels=50)

    plt.title('Consistency matrix')

    plt.colorbar()

    plt.show()



def compute_consistency_metrics(src_points, ref_points,

                                resolution: float = 1,

                                return_matrix: bool = True,

                                plot: bool = False) -> dict:

    """Compute the consistency metrics at a specified meter resolution. The grid

    size will depend on the range of x and y values in the src and ref points, as

    well as the desirable resolution.



    Returns - a dictionary with the following keys (consistency, std_of_mean, std_of_points

    are computed using only for grid cells with at least one hit):

        - consistency: mean RMS consistency of the grid for all grid points with hit,

        - std_of_mean: mean standard deviation of mean-src and mean-ref in a grid cell, where

                       mean-src and mean-ref are the mean of all points falling onto that

                       grid cell.

        - std_of_points: mean standard deviation of all points falling onto a grid cell,

        - hit_by_both: % grid cells hit by both src and ref,

        - hit_by_one: % grid cells hit by at least one of src or ref,

        - consistency_matrix: optional, a 2D numpy array with the consistency values for each grid cell,

                              grids with no hits will have a value of 0.

    """

    src_tree = KDTree(src_points[:, :2])

    ref_tree = KDTree(ref_points[:, :2])



    # Get range of x and y

    max_x = min(np.max(src_points[:, 0]), np.max(ref_points[:, 0]))

    min_x = max(np.min(src_points[:, 0]), np.min(ref_points[:, 0]))

    max_y = min(np.max(src_points[:, 1]), np.max(ref_points[:, 1]))

    min_y = max(np.min(src_points[:, 1]), np.min(ref_points[:, 1]))



    num_rows = int((max_x - min_x) / resolution) + 1

    num_cols = int((max_y - min_y) / resolution) + 1



    consistency_metric = np.zeros((num_rows, num_cols))

    std_of_mean_metric = np.zeros((num_rows, num_cols))

    std_of_points_metric = np.zeros((num_rows, num_cols))

    hit_by_both = np.zeros((num_rows, num_cols))

    hit_by_one = np.zeros((num_rows, num_cols))



    queries, queries_tree = _construct_query_tree(min_x=min_x,

                                                  min_y=min_y,

                                                  num_rows=num_rows,

                                                  num_cols=num_cols,

                                                  resolution=resolution)



    std_src = queries_tree.query_ball_tree(src_tree, resolution * .5)

    std_ref = queries_tree.query_ball_tree(ref_tree, resolution * .5)



    consistency_src = queries_tree.query_ball_tree(src_tree, resolution * 1.5)

    consistency_ref = queries_tree.query_ball_tree(ref_tree, resolution * 1.5)



    for i, query in enumerate(queries):

        row = int((query[0] - min_x) / resolution)

        col = int((query[1] - min_y) / resolution)

        hits_consistency_src = consistency_src[i]

        hits_consistency_ref = consistency_ref[i]

        std_src_idx = std_src[i]

        std_ref_idx = std_ref[i]



        maxmin_dist_src_to_ref, maxmin_dist_ref_to_src = 0, 0

        if len(hits_consistency_src) > 0 and len(std_ref_idx) > 0:

            maxmin_dist_ref_to_src = np.min(cdist(

                ref_points[std_ref_idx], src_points[hits_consistency_src]),

                                            axis=1).max()

        if len(hits_consistency_ref) > 0 and len(std_src_idx) > 0:

            maxmin_dist_src_to_ref = np.min(cdist(

                src_points[std_src_idx], ref_points[hits_consistency_ref]),

                                            axis=1).max()

        consistency_metric[row, col] = np.max(

            [maxmin_dist_src_to_ref, maxmin_dist_ref_to_src])



        if len(std_src_idx) == 0 and len(std_ref_idx) == 0:

            continue

        std_of_points_metric[row, col] = np.std(

            np.concatenate((src_points[std_src_idx, 2], ref_points[std_ref_idx,

                                                                   2])))

        if len(std_src_idx) > 0 and len(std_ref_idx) > 0:

            std_of_mean_metric[row, col] = np.std([

                np.mean(src_points[std_src_idx, 2]),

                np.mean(ref_points[std_ref_idx, 2])

            ])

        hit_by_both[row, col] = len(std_src_idx) > 0 and len(std_ref_idx) > 0

        hit_by_one[row, col] = len(std_src_idx) > 0 or len(std_ref_idx) > 0



    mean_of_grid_with_values = lambda grid: np.mean(grid[grid > 0]) if np.sum(

        grid > 0) > 0 else 0



    results = {

        'consistency': mean_of_grid_with_values(consistency_metric),

        'std_of_mean': mean_of_grid_with_values(std_of_mean_metric),

        'std_of_points': mean_of_grid_with_values(std_of_points_metric),

        'hit_by_both': hit_by_both.mean(),

        'hit_by_one': hit_by_one.mean(),

    }

    for key, value in results.items():

        print(f'{key}: {value}')

    if return_matrix:

        results['consistency_matrix'] = consistency_metric

    if plot:

        plot_consistency_matrix(consistency_metric)

    return results



def parse_args():

    parser = ArgumentParser()

    parser.add_argument('--src', type=str, required=True, help='Path to the source points (npy file).')

    parser.add_argument('--ref', type=str, required=True, help='Path to the reference points (npy file).')

    parser.add_argument('--resolution', type=float, default=1, help='Resolution of the grid (meters).')

    parser.add_argument('--plot', action='store_true', help='Plot the consistency matrix.')

    return parser.parse_args()



def main():

    args = parse_args()

    src_points = np.load(args.src)

    ref_points = np.load(args.ref)

    compute_consistency_metrics(src_points, ref_points, resolution=args.resolution, plot=args.plot)



if __name__ == '__main__':

    main()