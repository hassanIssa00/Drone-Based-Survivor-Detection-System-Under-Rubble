#!/usr/bin/env python3
"""
Building Collapse Controller
Simulates building collapse by applying forces to the structure
"""

import rospy
from gazebo_msgs.srv import ApplyBodyWrench, GetModelState, GetLinkState
from gazebo_msgs.msg import ModelState, LinkState
from geometry_msgs.msg import Wrench, Vector3, Point
import time

class BuildingCollapseController:
    def __init__(self):
        rospy.init_node('building_collapse_controller', anonymous=True)
        
        # Wait for Gazebo services
        rospy.wait_for_service('/gazebo/apply_body_wrench')
        rospy.wait_for_service('/gazebo/get_model_state')
        
        self.apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        
        self.building_name = 'building_to_collapse'
        
        rospy.loginfo("Building Collapse Controller Initialized")
    
    def trigger_collapse(self, delay=5.0):
        """
        Trigger building collapse by applying forces to structural components
        """
        rospy.loginfo(f"Building collapse will begin in {delay} seconds...")
        rospy.sleep(delay)
        
        rospy.logwarn("=" * 80)
        rospy.logwarn("BUILDING COLLAPSE INITIATED!")
        rospy.logwarn("=" * 80)
        
        # Define the collapse sequence
        collapse_sequence = [
            # (link_name, force_vector, duration)
            ('column1', Vector3(x=5000, y=2000, z=-3000), 0.5),
            ('column2', Vector3(x=-5000, y=2000, z=-3000), 0.5),
            ('column3', Vector3(x=5000, y=-2000, z=-3000), 0.5),
            ('column4', Vector3(x=-5000, y=-2000, z=-3000), 0.5),
            ('roof', Vector3(x=0, y=0, z=-8000), 1.0),
            ('floor1', Vector3(x=0, y=0, z=-5000), 1.0),
            ('wall1_fl', Vector3(x=3000, y=-2000, z=-2000), 0.3),
            ('wall1_fr', Vector3(x=-3000, y=-2000, z=-2000), 0.3),
            ('wall1_back', Vector3(x=0, y=3000, z=-2000), 0.3),
            ('wall1_left', Vector3(x=2000, y=0, z=-2000), 0.3),
            ('wall1_right', Vector3(x=-2000, y=0, z=-2000), 0.3),
        ]
        
        # Apply forces in sequence
        for link_name, force_vector, duration in collapse_sequence:
            full_link_name = f"{self.building_name}::{link_name}"
            
            try:
                rospy.loginfo(f"Applying collapse force to {link_name}...")
                
                # Create wrench
                wrench = Wrench()
                wrench.force = force_vector
                wrench.torque = Vector3(x=0, y=0, z=0)
                
                # Apply the wrench
                success = self.apply_wrench(
                    body_name=full_link_name,
                    reference_frame='world',
                    reference_point=Point(0, 0, 0),
                    wrench=wrench,
                    start_time=rospy.Time(0),
                    duration=rospy.Duration(duration)
                )
                
                if success:
                    rospy.loginfo(f"Force applied to {link_name} successfully")
                else:
                    rospy.logwarn(f"Failed to apply force to {link_name}")
                
                # Small delay between forces for realistic collapse
                rospy.sleep(0.2)
                
            except Exception as e:
                rospy.logerr(f"Error applying force to {link_name}: {e}")
        
        rospy.logwarn("=" * 80)
        rospy.logwarn("BUILDING COLLAPSE COMPLETE!")
        rospy.logwarn("Rescue operations can now begin...")
        rospy.logwarn("=" * 80)
    
    def run(self):
        """Main execution"""
        try:
            # Start collapse sequence
            self.trigger_collapse(delay=3.0)
            
            rospy.loginfo("Collapse simulation complete. Node will continue running...")
            rospy.spin()
            
        except rospy.ROSInterruptException:
            rospy.loginfo("Building collapse controller shutting down")

if __name__ == '__main__':
    try:
        controller = BuildingCollapseController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
