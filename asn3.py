#!/usr/bin/env python

#import roslib
#import rospy
#from fw_wrapper.srv import *
import time
import sys
import signal
import csv
import random
import math
import numpy as np

# -----------SERVICE DEFINITION-----------
# allcmd REQUEST DATA
# ---------
# string command_type
# int8 device_id
# int16 target_val
# int8 n_dev
# int8[] dev_ids
# int16[] target_vals

# allcmd RESPONSE DATA
# ---------
# int16 val
# --------END SERVICE DEFINITION----------

# ----------COMMAND TYPE LIST-------------
# GetMotorTargetPosition
# GetMotorCurrentPosition
# GetIsMotorMoving
# GetSensorValue
# GetMotorWheelSpeed
# SetMotorTargetPosition
# SetMotorTargetSpeed
# SetMotorTargetPositionsSync
# SetMotorMode
# SetMotorWheelSpeed

# wrapper function to call service to set a motor mode
# 0 = set target positions, 1 = set wheel moving
def setMotorMode(motor_id, target_val):
    rospy.wait_for_service('allcmd')
    try:
        send_command = rospy.ServiceProxy('allcmd', allcmd)
        resp1 = send_command('SetMotorMode', motor_id, target_val, 0, [0], [0])
        return resp1.val
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

# wrapper function to call service to get motor wheel speed
def getMotorWheelSpeed(motor_id):
    rospy.wait_for_service('allcmd')
    try:
        send_command = rospy.ServiceProxy('allcmd', allcmd)
        resp1 = send_command('GetMotorWheelSpeed', motor_id, 0, 0, [0], [0])
        return resp1.val
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

# wrapper function to call service to set motor wheel speed
def setMotorWheelSpeed(motor_id, target_val):
    rospy.wait_for_service('allcmd')
    try:
        send_command = rospy.ServiceProxy('allcmd', allcmd)
        resp1 = send_command('SetMotorWheelSpeed', motor_id, target_val, 0, [0], [0])
        return resp1.val
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

# wrapper function to call service to set motor target speed
def setMotorTargetSpeed(motor_id, target_val):
    rospy.wait_for_service('allcmd')
    try:
        send_command = rospy.ServiceProxy('allcmd', allcmd)
        resp1 = send_command('SetMotorTargetSpeed', motor_id, target_val, 0, [0], [0])
        return resp1.val
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

# wrapper function to call service to get sensor value
def getSensorValue(port):
    rospy.wait_for_service('allcmd')
    try:
        send_command = rospy.ServiceProxy('allcmd', allcmd)
        resp1 = send_command('GetSensorValue', port, 0, 0, [0], [0])
        return resp1.val
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

# wrapper function to call service to set a motor target position
def setMotorTargetPositionCommand(motor_id, target_val):
    rospy.wait_for_service('allcmd')
    try:
        send_command = rospy.ServiceProxy('allcmd', allcmd)
	resp1 = send_command('SetMotorTargetPosition', motor_id, target_val, 0, [0], [0])
        return resp1.val
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

# wrapper function to call service to get a motor's current position
def getMotorPositionCommand(motor_id):
    rospy.wait_for_service('allcmd')
    try:
        send_command = rospy.ServiceProxy('allcmd', allcmd)
	resp1 = send_command('GetMotorCurrentPosition', motor_id, 0, 0, [0], [0])
        return resp1.val
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

# wrapper function to call service to check if a motor is currently moving
def getIsMotorMovingCommand(motor_id):
    rospy.wait_for_service('allcmd')
    try:
        send_command = rospy.ServiceProxy('allcmd', allcmd)
	resp1 = send_command('GetIsMotorMoving', motor_id, 0, 0, [0], [0])
        return resp1.val
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

# shutdown function to stop wheels
def shutdown(sig, stackframe):
    print("  Caught ctrl-c!")
    setMotorWheelSpeed(5, 0)
    setMotorWheelSpeed(6, 0)
    sys.exit(0)    

def stop():
    setMotorWheelSpeed(5, 0)
    setMotorWheelSpeed(6, 0)


def loadData(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
        random.shuffle(dataset)
        trainingSet = dataset[:64]
        testSet = dataset[64:]
	return trainingSet, testSet

def distance(point1, point2, length):
    distance = 0
    for i in range(length):
        distance += pow((point1[i] - point2[i]), 2)
    return math.sqrt(distance)

def gaussian_kernel(x, x0, c, a = 1.0):
    diff = np.linalg.norm(x - x0)
    return a * np.exp(diff / (-2.0 * c**2))

def get_weights(training_inputs, datapoint, c = 1.0):
    x = np.mat(training_inputs)
    n_rows = x.shape[0]
    weights = np.mat(np.eye(n_rows))
    for i in xrange(n_rows):
        weights[i, i] = gaussian_kernel(datapoint, x[i], c)
    return weights

def lwr_predict(training_inputs, training_outputs, datapoint, c = 1.0):
    weights = get_weights(training_inputs, datapoint, c = c)

    x = np.mat(training_inputs)
    y = np.mat(training_outputs)

    xt = x.T * (weights * x)
    betas = xt.I * (x.T * (weights * y))

    return datapoint * betas

    
# Main function
if __name__ == "__main__":
    
    # taking command line arguments
    if len(sys.argv) > 1:
        i = int(sys.argv[1])
        j = int(sys.argv[2])
        c = float(sys.argv[3])

    '''
    # node startup and signal declaration
    rospy.init_node('example_node', anonymous=True)
    rospy.loginfo("Starting Group X Control Node...")
    signal.signal(signal.SIGINT, shutdown)
    
    # setting motors to wheel mode
    setMotorMode(5, 1)
    setMotorMode(6, 1)
    '''
    # Python command line commands 
    x, y = loadData("arcData.csv", 1.0)
    train = np.mat(x)
    train_input = train[:,2:]
    train_output = train[:,:2]
    target = np.mat([[int(i),int(j)]])
    print target
    predicted_output = lwr_predict(train_input, train_output, target, c=c)

    print predicted_output[0,0], predicted_output[0,1]

    setMotorWheelSpeed(5, 1024 + predicted_output[0,1])
    setMotorWheelSpeed(6, predicted_output[0,0])
    time.sleep(8.0)
    setMotorWheelSpeed(5, 0)
    setMotorWheelSpeed(6, 0)

    #train-test split validation
    for i in range(5):
        train, test = loadData("arcData.csv", 0.8)
        train_m = np.mat(train)
        train_input = train_m[:,2:]
        train_output = train_m[:,:2]

        ith_collected = []
        for j in range(len(test)):
            target = np.mat([int(test[j][2]), int(test[j][3])])
            model_output = lwr_predict(train_input, train_output, target, c = 0.8)
            distance = np.sqrt( (model_output[0,0] - test[j][0])**2 + (model_output[0,1] - test[j][1])**2)
            ith_collected.append(distance)
            print 'Distance this trial: ', distance
        print "Distance over test: ",  sum(ith_collected)/len(ith_collected)



    '''
    trainingSet = []
    testSet = []
    trainingSet, testSet = loadData(str(data), 0.8, trainingSet, testSet)
    print 'Train: ' + repr(len(trainingSet))
    print 'Test: ' + repr(len(testSet))
    '''


    '''
    # data collection process
    left = random.randrange(400, 700, 1)
    print 'Random left wheel speed: ', left
    right = random.randrange(400, 700, 1)
    print 'Random right wheel speed: ', right
    setMotorWheelSpeed(5, 1024 + right)
    setMotorWheelSpeed(6, left)
    time.sleep(8.0)
    setMotorWheelSpeed(5, 0)
    setMotorWheelSpeed(6, 0)
    
    
    # control loop running at 10hz
    r = rospy.Rate(10)# 10hz
    
    while not rospy.is_shutdown():
        
        reading_front = getSensorValue(3)
        reading_left = getSensorValue(1)
        reading_right = getSensorValue(6)
        rospy.loginfo("Front port: %f    Left Port: %f    Right Port: %f", \
                reading_front, reading_left, reading_right)
        
        
        # Sleep to enforce loop rate
        r.sleep()
    '''
    
    




