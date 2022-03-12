## Accuracy is 95.32% all rooms , task_idx = 0,1,2,3, trial = 0 for TP, FP = 1366 67 sentences

#some examples of FN that does not make sense:
# sentence: look left
# ground truth target = toiletpaperhanger!
# Our prediction for target object = None
#or
# sentence: turn around and take one step then turn left and look up
# ground truth target = handtowelholder!
# Our prediction: None

# For Carry we considered the object which the robot is carried as a target object 


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
import pandas as pd
import re

os.environ['MAIN'] = '../'
sys.path.append(os.path.join(os.environ['MAIN']))
from robot.sensing import sensing
from language_understanding import instruction_parser as ip
import mapper.params as params

env = sensing()
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
    event = env.step(dict({"action": "GetReachablePositions", 'renderClassImage': True}))
    #event = env.step(dict(traj_data['scene']['init_action']))

    return env,event,traj_data

def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt

def equivalent_words(word):

    o = word
    objs = ""

    if o=="tissues":
        objs = objs+"TissueBox"+','
    if o=="counter" or o=="island" or o=="islandcounter" or o=="kitchenisland" or o =="cabinet" or o =="cabinets" or o=="microwave" or o == "kitchencounter" or o=="sinkcounter":
        objs = objs +"CounterTop"+','
    if o=="cupboard":
        objs = objs +"Cabinet"+','
    if o=="coffeemaker" or o=="coffeemachine" or o=="coffee":
        objs = objs +"CoffeeMachine"
    if o=="stove" or o=="oven" or o=="range" or o=="Oven":
        objs = objs +"Stove"+','
    if o=="stool" or o=="footstool":
        objs = objs+"Ottoman"+','
    if o=="bottle" or o=="Bottle" or o=="container":
        objs = objs+"Vase"+','
    if o=="tray":
        objs = objs+"Plate"+','
    if o=="rag":
        objs = objs+"Cloth"+','
    if o=="computer" or o=="compute":
        objs = objs+"Laptop"+','
    if o=="bureau":
        objs = objs+"Bureau"+','
    if o=="clock":
        objs = objs+"AlarmClock"+','
    if o=="bookshelves" or o=="shelving" or o=="deskshelf" or o=="bookcase" or o=="desk" or o =="shelves" or o=="table" or o=="ledge":
        objs = objs+"Shelf"+','
    if o=="disc" or o=="discs" or o=="cd" or o == "disk":
        objs = objs+"CD"+','
    if o=="card":
        objs = objs+"CreditCard"+','
    if o=="refrigerator":
        objs = objs+"Fridge"+','
    if o=="saltshaker":
        objs = objs+"PepperShaker"+','
    if o=="counter":
        objs = objs+"CounterTop"+','
    if o=="keys":
        objs = objs+"KeyChain"+','
    if o=="bat" or o=="paddle":
        objs = objs+"BaseballBat"+','
    if o=="bean":
        objs = objs+"Beanbag"+','
    if o=="goblet":
        objs = objs+"Cup"+','
    if o=="table":
        objs = objs+"Desk"+','
    if o=="remotecontroller":
        objs = objs+"RemoteControl"+','
    if o=="drawers":
        objs = objs+"Dresser"+','
    if o=="dresserdrawer": #happens in 306-8
        objs = objs+"Drawer"+','
    
    if o=="Lamp" or o=="lamp" or o=="light" or o=="tablelamp":
        objs = objs+"DeskLamp"+','
        objs = objs+"FloorLamp"+','

    if o=="nightstand" or o=="Nightstand" or o=="side-table" or o=="sidetable" or o=="bedsidetable" or o=="nighttable" or o =="stand" or o == "dresser":
        objs = objs+"SideTable"+','
    
    if o=="trashcan" or o=="trash" or o=="Garbage" or o=="garbage" or o=="garbagebin" or o=="trashbin" or o=="trashcanister" or o=="recycle" or o=="recycling" or o=="recyclingbin" or o=="bin":
        objs = objs+"GarbageCan"+','

    if o=="cellphone" or o=="Cellphone" or o=="cell" or o=="Cell" or o=="phone" or o=="Phone" or o=="i-phone":
        objs = objs+"CellPhone"+','
    if o=="plant" or o=="Plant":
        objs = objs+"HousePlant"+','
    if o=="basket" or o=="Basket":
        objs = objs+"LaundryHamper"+','
    if o=="racket" or o=="Racket" or o=="tennisracket" or o=="paddle":
        objs = objs+"TennisRacket"+','
    if o=="tv" or o=="Television" or o=="television" or o=="TV" or o=="tvstand":
        objs = objs+"Television"+','
    if o=="tvstand" or o=="TVstand" or o=="TVStand":
        objs = objs+"TVStand"+','
    if o=="dish":
        objs = objs+"Bowl"+','
    if o=="couch" or o=="Couch":
        objs = objs+"Sofa"+','
    if o=="sponge":
        objs = objs+"DishSponge"+','
    if o=="teakettle":
        objs = objs+"Kettle"+','
    if o=="door" or o=="Door":
        objs = objs+"StandardDoor"+','
    if o=="coffeetable":
        objs = objs+"CoffeeTable"+','
    if o == "jug" or o=="vase":
        objs = objs+"GlassBottle"+','

    return objs


p = ip.parse() #takes around 0.05s for parsing a single sentence

TP = 0
FP = 0

#for room_range in [[301,302]]: ### debugging 
for room_range in [[301,331], [201,231], [1,31], [401,431]]:

    for r in range(room_range[0], room_range[1]):
        for task_idx in [0,1,2,3]:
        #task_idx = 0
            trial = 0
            try:
            	print(r)
            	traj_file = get_file(rn = r, task_index = task_idx, trial_num = trial) #take default first task of each room
            except:
                print("File is not present ")
                continue

            env,event,traj_data = set_env(traj_file, env = env)
            #print(traj_data)
            sentences = traj_data['turk_annotations']['anns'][0]["high_descs"]
            #print("Sentences =",sentences)
            list_intent, list_dic_parsing = p.predict(sentences)
            #print(list_intent)
            #print(list_dic_parsing)

            gt_targets = []
            pr_targets = []
            room_pre_gt = {}
            for i in range(len(list_dic_parsing)):
                sentence_equi_words = []
                sentence = re.sub(r'[^\w\s]', '', sentences[i]).lower()
                #print("sentence after premoving =", sentence)
                sent_words = sentence.split(" ")
                #print(sent_words)
                for word in sent_words:
                    equi_words = equivalent_words(word)[:-1]
                    equi_word_list = equi_words.split(",")
                    #print(equi_word_list)
                    for eq_word in equi_word_list:
                        sentence_equi_words.append(eq_word.lower())
                print("equivalent objects for each word of sentence based on our dictionary =", sentence_equi_words)



                target = traj_data['plan']['high_pddl'][i]['discrete_action']['args']
                if target != []:
                    gt_target = target[0]
                    gt_targets.append(target[0])
                    print("ground truth target =", gt_target)

                ### CHECK if we have target object key in prediction ###
                #if gt_target in sentence_equi_words:
                if "target_obj" in list(list_dic_parsing[i].keys()):
                    pr_target = list_dic_parsing[i]["target_obj"]
                    print("predicted target =", pr_target)
                else:
                    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& False &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                    FP+=1
                    continue


                print("equivalent_target =",[equivalent.lower() for equivalent in equivalent_words(pr_target).split(",")])
                if isinstance(pr_target, str):
                    #print("equivalent_target =", equivalent_words(pr_target).split(","))
                    if pr_target in gt_target:
                        TP+=1
                    else:
                        equivalents = equivalent_words(pr_target).split(",")
                        if gt_target in [equivalent.lower() for equivalent in equivalents]:
                            TP+=1
                        else:
                            if gt_target not in sentence_equi_words:
                                continue
                            else:
                                print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& False &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                                FP+=1
                else:
                    flatten_target = flatten(pr_target)
                    print("flat_target is:", flatten_target)
                    TP_inloop=0
                    for pr_target in flatten_target:
                        print("predicted object from fltten list =", pr_target)
                        if pr_target in gt_target:
                            TP_inloop=1
                            break
                        else:
                            equivalents = equivalent_words(pr_target).split(",")
                            if gt_target in [equivalent.lower() for equivalent in equivalents]:
                            #if equivalent_words(pr_target).lower() == gt_target:
                                TP_inloop+=1
                                break
                            else:
                                if gt_target not in sentence_equi_words:
                                    TP_inloop= None

                    
                    if TP_inloop == 1:
                        TP+=1  
                    elif TP_inloop == None: 
                        break   
                    else:
                        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& False &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                        FP+=1

                print("TP, FP object =", TP, FP)  

    print("TP, FP =", TP, FP)
    Accuracy = TP/(TP+FP)
    print("Accuracy =", Accuracy)

            #print("Ground Truth Targets =", gt_targets)
                    