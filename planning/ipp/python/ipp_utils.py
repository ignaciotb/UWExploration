import numpy as np
import torch
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import rospy
from sensor_msgs import point_cloud2

def generate_local_bounds(global_bounds, vehicle_position, horizon_distance, safety_margin):
    """ To plan in a local area rather than the entire survey area, this function can generate bounds
        that the planner then uses. Since this function is used by several classes, it is deemed a
        utility function.

    Args:
        global_bounds    (list[double]): The absolute boundaries of the entire map, in [low_x, low_y, high_x, high_y]
        vehicle_position (list[double]): Vehicle position in [x y] (can be passed in with heading too)
        horizon_distance       (double): The distance from vehicle position to create local bounds (while inside global bounds)
        safety_margin          (double): A margin between the created local bounds and the global bounds

    Returns:
        (list[double]): The generated local bounds
    """   
    
    low_x           = max(global_bounds[0] + safety_margin, min(vehicle_position[0] - horizon_distance, vehicle_position[0] + horizon_distance))
    low_y           = max(global_bounds[1] + safety_margin, min(vehicle_position[1] - horizon_distance, vehicle_position[0] + horizon_distance))
    high_x          = min(global_bounds[2] - safety_margin, max(vehicle_position[0] - horizon_distance, vehicle_position[0] + horizon_distance))
    high_y          = min(global_bounds[3] - safety_margin, max(vehicle_position[1] - horizon_distance, vehicle_position[0] + horizon_distance))
    
    return [low_x, low_y, high_x, high_y]

def upsample_waypoints(pose1, pose2, resolution):
    """ Creates intermediate poses between two poses, at a specified resolution

    Args:
        pose1 (list[double]): Start waypoint [x y theta]
        pose2 (list[double]): End waypoint [x y theta]
        resolution  (double): Distance between upsampled waypoints

    Returns:
        list[double]: Waypoints [[x y theta],...]
    """
    n = round(np.hypot(pose1[0]-pose2[0], pose1[1]-pose2[1])/resolution)
    angle = np.arctan2(pose2[1]-pose1[1],pose2[0]-pose1[0])
    poses = []
    dx = resolution*np.cos(angle)
    dy = resolution*np.sin(angle)
    for i in range(n):
        poses.append([pose1[0]+i*dx, pose1[1]+i*dy, angle])
    return poses
        
def get_orthogonal_samples(poses, nbr_samples=10, swath_width=20):
    """ Generates points on lines orthogonal to a vector. Will generate
        `nbr_samples` for each given vector, along a line of given swath width.
    
    Args:
        poses (list[float]): [x y theta]
        nbr_samples (int, optional): number of samples generated for each vector. Defaults to 10.
        swath_width (float, optional): width of line sampled from, for each vector. Defaults to 20.

    Returns:
        np.array: concatenated xy points of samples with dimensions [len(poses) * nbr_samples, 2]
    """
    samples = np.zeros((len(poses), nbr_samples, 2))
    radius = 0.5*swath_width
    poses = np.array(poses)
    x = poses[:,0]
    y = poses[:,1]
    direction = poses[:,2] + np.pi/2
    dx = radius*np.cos(direction) 
    dy = radius*np.sin(direction) 

    op = np.expand_dims(np.linspace(-1, 1, nbr_samples), 0)
    dx_s = np.expand_dims(dx, 1) * op + np.expand_dims(x, 1)
    dy_s = np.expand_dims(dy, 1) * op + np.expand_dims(y, 1)
    samples[:,:,0] = dx_s 
    samples[:,:,1] = dy_s
    output = np.reshape(samples, (len(poses) * nbr_samples, 2))
    return output #torch.from_numpy(output).type(torch.FloatTensor)

def save_model(model, filename):
        """ Saves a torch model

        Args:
            model (): Given torch model to save
            filename (string): Location to save model
        """
        torch.save({'model' : model.state_dict()}, filename)
    
def load_model(model, filename):
    """ Loads a torch model from state dictionary

    Args:
        model (): Given model to load state into
        filename (string): Location of the saved state dictionary
    """
    cp = torch.load(filename)
    model.load_state_dict(cp['model'])
    return model

def pack_cloud(frame, mbes):
    mbes_pcloud = PointCloud2()
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1)]

    mbes_pcloud = point_cloud2.create_cloud(header, fields, mbes)

    return mbes_pcloud