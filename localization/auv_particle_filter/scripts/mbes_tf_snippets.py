#!/usr/bin/env python 

"""
NOTE: This is not a script that runs

These snippets of code were used to broadcast and listen to tf2 transforms
between particles, their respective mbes liinks, and the map (odom) frame

They have been removed from the auv_pf.py node for a more efficient alternative,
but are being saved here for future reference

The class and function they were a part of has been included for context on where they were used
"""

# Initialize tf transform to each particle and mbes_link
"""
class Particle():
    def __init__():
        self.frame_id = "particle_" + str(index)
        self.mbes_frame_id = "particle_" + str(index) + "_mbes_link"
        
        self.transform = TransformStamped() 
        self.transform.header.frame_id = map_frame
        self.transform.child_frame_id = self.frame_id
        self.transform.transform.rotation.w = 1 
        broadcaster.sendTransform(self.transform)

        self.mbes_trans = TransformStamped()
        self.mbes_trans.transform = mbes_trans.transform
        self.mbes_trans.header.frame_id = self.frame_id
        self.mbes_trans.child_frame_id = self.mbes_frame_id
        self.mbes_trans.header.stamp = rospy.Time.now()
        static_broadcaster.sendTransform(self.mbes_trans)
"""

# Update tf to each particle during prediction steps
"""
class Particle():
    def update():
        
        self.transform.transform.translation = self.pose.position
        self.transform.transform.rotation = self.pose.orientation
        self.transform.header.stamp = rospy.Time.now()
        broadcaster.sendTransform(self.transform)
"""

# Listen to tf transform from particle mbes to map
# For sending to mbes sim action server
"""
class auv_pf():
    def pf2mbes():
        
        trans = self.tfBuffer.lookup_transform(self.map_frame, particle_.mbes_frame_id , rospy.Time())
"""