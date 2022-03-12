### Generate depth images of four rotations (front, right, left, and back)
import numpy as np
import math
import sys
import glob
import os
import json
import random
import copy
import argparse
import cv2
import sys
import json

#sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
#sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))

#from env.thor_env import ThorEnv

os.environ['MAIN'] = '../'
sys.path.append(os.path.join(os.environ['MAIN']))
from robot.sensing import sensing
import mapper.params as params


def get_file(rn = 302, task_index = 1, trial_num = 0):

    folders = sorted(glob.glob(params.trajectory_data_location+repr(rn))) #for home computer
    print("Number of demonstrated tasks for this room ",len(folders))
    trials = glob.glob(folders[task_index]+'/*') #there would be len(folders) number of different tasks 
    print("Number of different trials (language instr) for the same task ",len(trials))
    traj = glob.glob(trials[trial_num]+'/*.json')
    print("got trajectory file ",traj)
    return traj

def set_env(json_file, env = []):
    if env==[]:
        env = sensing()
    
    with open(json_file[0]) as f:
        traj_data = json.load(f)
    #print("loaded traj file")
    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    object_toggles = traj_data['scene']['object_toggles']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']

    # reset
    scene_name = 'FloorPlan%d' % scene_num
    print("ROOM NUMBER ", scene_name)
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    print("setting orientation of the agent to facing north ")
    traj_data['scene']['rotation'] = 0
    ### Get the reachable positions in the room
    event = env.step(dict({"action": "GetReachablePositions"}))
    #event = env.step(dict(traj_data['scene']['init_action']))

    return env,event,traj_data

def xyz_reachable_pos(i_th_pose,reach_pos):
    x_pos = reach_pos[i_th_pose]["x"]
    y_pos = reach_pos[i_th_pose]["y"]
    z_pos = reach_pos[i_th_pose]["z"]
    return x_pos,y_pos,z_pos

def save_images(env, event, img_folder, depth_folder, floor_plan, x_pos, y_pos, z_pos, orientation):
    rgb_image = event.frame[:, :, ::-1]
    cv2.imwrite(img_folder + "img_{floor_plan}_x_{i1}_y_{i2}_z_{i3}_{i4}.png".format(floor_plan = floor_plan, i1= x_pos, i2= y_pos, i3= z_pos, i4 = orientation) ,rgb_image)

    depth = env.get_depth_image() * (255 / 10000)
    depth = depth.astype(np.uint8)
    cv2.imwrite(depth_folder + "depth_{floor_plan}_x_{i1}_y_{i2}_z_{i3}_{i4}.png".format(floor_plan = floor_plan, i1= x_pos, i2= y_pos, i3= z_pos, i4 = orientation) ,depth)
    return

def rotate(Rotate_des,env): # Rotate_desc = "RotateRight"
    return env.step(dict({"action": Rotate_des}))
#IMAGE_WIDTH = 300 #rendering
#IMAGE_HEIGHT = 300
#env = ThorEnv(player_screen_width=IMAGE_WIDTH,player_screen_height=IMAGE_HEIGHT) #blank ai2thor environment



if __name__ == '__main__':

    room_range = [401,431]
    print("creating all the images and depth images")
    img_folder = 'images/Training/RGB/bathroom/'
    depth_folder = 'images/Training/depth/bathroom/'
    generating_images = False
    generating_txt_file = True

    if generating_images:
        env = sensing()

        for r in range(room_range[0], room_range[1]):
            try:
                traj_file = get_file(rn = r, task_index = 0, trial_num = 0) #take default first task of each room
            except:
                print("File is not present ")
                continue
            env,event,traj_data = set_env(traj_file, env = env)

            reach_pos = event.metadata["actionReturn"]
            print("Numbers of reachable positions in room =", len(reach_pos))
            some_reach_pos = random.sample(reach_pos,20)
            for i in range(len(some_reach_pos)):
                x_pos,y_pos,z_pos = xyz_reachable_pos(i,reach_pos)
                print(x_pos)
                event = env.step(dict({"action":"Teleport", "x": x_pos, 'y': y_pos, 'z': z_pos}))
                save_images(env, event, img_folder, depth_folder, r, x_pos,y_pos,z_pos,"Front")
                event = rotate("RotateRight",env)
                save_images(env, event, img_folder, depth_folder, r, x_pos,y_pos,z_pos,"Right")
                event = rotate("RotateRight",env)
                save_images(env, event, img_folder, depth_folder, r, x_pos,y_pos,z_pos,"Back")
                event = rotate("RotateRight",env)
                save_images(env, event, img_folder, depth_folder, r, x_pos,y_pos,z_pos,"Left")

    # writing text file which in each line of this text file we have the path ro rgb image, the path to corresponding depth image and focal length
    if generating_txt_file:

        txt_file = open(r"alfred_rgb_depth.txt","w+")
        for dir in os.listdir('images/Training/RGB/'):
            for file in os.listdir('images/Training/RGB/' + dir):
                first_element = '/'+str(dir)+'/'+str(file)
                second_element = '/'+str(dir)+'/depth'+str(file[3:])
                third_element = str(518.8579)
                list_line = first_element+ ' ' + second_element + ' ' + third_element
                txt_file.write(list_line)
                txt_file.write('\n')
                #print(first_element)
                #print(second_element)



