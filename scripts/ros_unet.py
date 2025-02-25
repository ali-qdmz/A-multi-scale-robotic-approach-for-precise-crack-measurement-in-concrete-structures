#!/usr/bin/env python3
import math
import torch
import torchvision
import torch.nn as nn
import albumentations as A
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from albumentations.pytorch import ToTensorV2 
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
import tf


import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
# import tensorflow as tf
from std_msgs.msg import String


from move_it import *

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


def double_conv(in_ch, out_ch):
    conv = nn.Sequential(
        nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(out_ch),                                                            
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1), 
        nn.BatchNorm2d(out_ch),                                                            
        nn.ReLU(inplace=True)
    )
    
    return conv

#def cropper(og_tensor, target_tensor):
#    og_shape = og_tensor.shape[2]
#    target_shape = target_tensor.shape[2]
#    delta = (og_shape - target_shape) // 2
#    cropped_og_tensor = og_tensor[:,:,delta:og_shape-delta,delta:og_shape-delta]
#    return cropped_og_tensor
 
    
def padder(left_tensor, right_tensor): 
    # left_tensor is the tensor on the encoder side of UNET
    # right_tensor is the tensor on the decoder side  of the UNET
    
    if left_tensor.shape != right_tensor.shape:
        padded = torch.zeros(left_tensor.shape)
        padded[:, :, :right_tensor.shape[2], :right_tensor.shape[3]] = right_tensor
        return padded.to(CFG.device)
    
    return right_tensor.to(CFG.device)




class CFG:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_pct = 0.2
    learning_rate = 3e-4
    batch_size = 4
    epochs = 3


class UNET(nn.Module):
    def __init__(self,in_chnls, n_classes):
        super(UNET,self).__init__()
        
        self.in_chnls = in_chnls
        self.n_classes = n_classes
        
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.down_conv_1 = double_conv(in_ch=self.in_chnls,out_ch=64)
        self.down_conv_2 = double_conv(in_ch=64,out_ch=128)
        self.down_conv_3 = double_conv(in_ch=128,out_ch=256)
        self.down_conv_4 = double_conv(in_ch=256,out_ch=512)
        self.down_conv_5 = double_conv(in_ch=512,out_ch=1024)
        #print(self.down_conv_1)
        
        self.up_conv_trans_1 = nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=2,stride=2)
        self.up_conv_trans_2 = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=2,stride=2)
        self.up_conv_trans_3 = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=2)
        self.up_conv_trans_4 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2)
        
        self.up_conv_1 = double_conv(in_ch=1024,out_ch=512)
        self.up_conv_2 = double_conv(in_ch=512,out_ch=256)
        self.up_conv_3 = double_conv(in_ch=256,out_ch=128)
        self.up_conv_4 = double_conv(in_ch=128,out_ch=64)
        
        self.conv_1x1 = nn.Conv2d(in_channels=64,out_channels=self.n_classes,kernel_size=1,stride=1)
        
    def forward(self,x):
        
        # encoding
        x1 = self.down_conv_1(x)
        #print("X1", x1.shape)
        p1 = self.max_pool(x1)
        #print("p1", p1.shape)
        x2 = self.down_conv_2(p1)
        #print("X2", x2.shape)
        p2 = self.max_pool(x2)
        #print("p2", p2.shape)
        x3 = self.down_conv_3(p2)
        #print("X2", x3.shape)
        p3 = self.max_pool(x3)
        #print("p3", p3.shape)
        x4 = self.down_conv_4(p3)
        #print("X4", x4.shape)
        p4 = self.max_pool(x4)
        #print("p4", p4.shape)
        x5 = self.down_conv_5(p4)
        #print("X5", x5.shape)
        
        # decoding
        d1 = self.up_conv_trans_1(x5)  # up transpose convolution ("up sampling" as called in UNET paper)
        pad1 = padder(x4,d1) # padding d1 to match x4 shape
        cat1 = torch.cat([x4,pad1],dim=1) # concatenating padded d1 and x4 on channel dimension(dim 1) [batch(dim 0),channel(dim 1),height(dim 2),width(dim 3)]
        uc1 = self.up_conv_1(cat1) # 1st up double convolution
        
        d2 = self.up_conv_trans_2(uc1)
        pad2 = padder(x3,d2)
        cat2 = torch.cat([x3,pad2],dim=1)
        uc2 = self.up_conv_2(cat2)
        
        d3 = self.up_conv_trans_3(uc2)
        pad3 = padder(x2,d3)
        cat3 = torch.cat([x2,pad3],dim=1)
        uc3 = self.up_conv_3(cat3)
        
        d4 = self.up_conv_trans_4(uc3)
        pad4 = padder(x1,d4)
        cat4 = torch.cat([x1,pad4],dim=1)
        uc4 = self.up_conv_4(cat4)
        
        conv_1x1 = self.conv_1x1(uc4)
        return conv_1x1
        #print(conv_1x1.shape)




class ImageProcessor:
    def __init__(self):
        # Initialize the node
        rospy.init_node('image_processor', anonymous=True)
        self.navigation_cordinates = None
        # Create a CVBridge object
        trained_model = UNET(in_chnls = 3, n_classes = 1)
        self.bridge = CvBridge()
        UNET_TRAINED = "/home/ali/unet_scratch.pth"
        trained_model.load_state_dict(torch.load(UNET_TRAINED))
        trained_model = trained_model.to("cuda")
        # Load your pretrained model
        self.model = trained_model
        self.depth_image_data = None
        # Subscribe to the realsense topic
        self.image_sub = rospy.Subscriber("/realsense/color/image_raw", Image, self.callback)
        self.depth_image_sub = rospy.Subscriber("/realsense/depth/image_rect_raw", Image, self.depth_callback)
        # Publisher for the results
        self.result_pub = rospy.Publisher('model_result_image', Image, queue_size=10)


    def depth_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "32FC1") # Convert the depth image to a Numpy array
        except CvBridgeError as e:
            print(e)
            
        self.depth_image_data = cv_image


    def img_to_3D(self, u, v):
        color = (0, 255, 0)
        thickness = 2
        # u, v = 500, 400
        # Intrinsic camera parameters 
        f = 350# Assumed focal length
        cx, cy = 319.5, 239.5 # Assumed optical center
        angle = math.pi/2 + 0.8

        T = np.array([
            [math.cos(angle), 0, math.sin(angle), 0.08], 
            [0, 1, 0, 0], 
            [-math.sin(angle), 0, math.cos(angle), 0.8], 
            [0, 0, 0, 1]
        ])

        Z = self.depth_image_data[v, u]
        X = -(u - cx) * Z / f 
        Y = (v - cy) * Z / f

        # Homogeneous coordinates
        p_camera_frame = np.array([Y, X, Z, 1])

        # Apply extrinsic matrix to transform point into world frame
        p_world_frame = T @ p_camera_frame
        # p_world_frame = T2 @ p_world_frame
        # p_world_frame = p_camera_frame

        # Normalize if w is not 1
        if p_world_frame[3] != 1:
            p_world_frame = p_world_frame / p_world_frame[3]

        return p_world_frame


    def calculate_distance(self, x, y, z):
        return math.sqrt(x**2 + y**2 + z**2)
    
    def callback(self, data):
        # Convert the image from a ROS msg to a CV image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        test_transform = A.Compose([A.Resize(572,572),
                                A.Normalize(mean=(0,0,0),std=(1,1,1),max_pixel_value=255),
                                ToTensorV2()])


        test_image = test_transform(image = cv_image)

        # print(test_image)

        # print(test_image["image"].dtype)
        # print(test_image["image"].shape)

        img = test_image["image"].unsqueeze(0)
        img = img.to("cuda")
        # Preprocess the image for your model (this will depend on the specific model you are using)
        # cv_image = cv2.resize(cv_image, (224, 224))  # Change to match model input
        # cv_image = cv_image / 255.0  # Normalization, if necessary
        # cv_image = cv_image.reshape(-1, 224, 224, 3)  # Reshape to match model input

        # Run the model on the image
        result = self.model(img)
        mask = result.squeeze(0).cpu().detach().numpy()
        # print(mask.shape)
        mask = mask.transpose(1,2,0)
        mask[mask < 0]=0
        mask[mask > 0]=1

        mask = np.squeeze(mask)

        mask = cv2.resize(mask, (cv_image.shape[1], cv_image.shape[0]), interpolation = cv2.INTER_NEAREST)
        mask = (mask * 255).astype(np.uint8)
        _, thresh = cv2.threshold(mask, 0, 255, 0)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Load original image
        # Replace 'image.png' with your input image file path or image array.







        u = 100
        v = 240


        pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=1)

        # header = Header()
        # header.stamp = rospy.Time.now()
        # header.frame_id = 'world' # Whatever your map frame is
        # p_world_frame = self.img_to_3D(u, v)
        # # Create a PointCloud2 message
        # points = [p_world_frame[:3]]
        # cloud = pc2.create_cloud_xyz32(header, points)

        # # Publish the PointCloud2 message
        # pub.publish(cloud)


        # cv2.circle(cv_image, (u, v), radius=5, color=(0, 0, 255), thickness=-1) # BGR color


        
        # bridge = CvBridge()
        # image_msg = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        # image_pub = rospy.Publisher('/image_with_red_dot', Image, queue_size=1)
        # Publish the image
        # image_pub.publish(image_msg)
        # print(f"The 3D position of the pixel in the world frame is: {p_world_frame[:3]}")

        # listener = tf.TransformListener()

        # (trans, rot) = listener.lookupTransform('/world', '/front_realsense', rospy.Time(0))
        # print(trans)
        # print(rot)
        # print("=====")
        # Draw bounding rectangles
        points = []
        navigation_cordinates = []
        for contour in contours:
            # compute the bounding box for the contour
            x, y, w, h = cv2.boundingRect(contour)
            start_target_crodinates = (round((x + w/2) - 1), y+h-10)
            end_target_crodinates = (round((x + w/2) - 1), y)
            check_cordinates = self.img_to_3D(start_target_crodinates[0], start_target_crodinates[1])
            check_cordinates = check_cordinates[:3]
            if w * h > 1200 and self.calculate_distance(check_cordinates[0], check_cordinates[1], check_cordinates[2]) <= 0.5:
                points_3D = []
                cv2.rectangle(cv_image, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)
                cv2.circle(cv_image, (start_target_crodinates[0], start_target_crodinates[1]), radius=5, color=(0, 255, 0), thickness=-1) # BGR color
                # cv2.circle(cv_image, (end_target_crodinates[0], end_target_crodinates[1]), radius=5, color=(0, 255, 0), thickness=-1) # BGR color
                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = 'world' # Whatever your map frame is

                p_world_frame = self.img_to_3D(start_target_crodinates[0], start_target_crodinates[1])
                points_3D.append(p_world_frame[:3])
                points.append(p_world_frame[:3])

                p_world_frame = self.img_to_3D(end_target_crodinates[0], end_target_crodinates[1])
                if self.calculate_distance(p_world_frame[0], p_world_frame[1], p_world_frame[2]) <= 0.5:
                    points_3D.append(p_world_frame[:3])
                    points.append(p_world_frame[:3])
                else:
                    i = 1
                    while self.calculate_distance(p_world_frame[0], p_world_frame[1], p_world_frame[2]) > 0.4:
                        end_target_crodinates = (round((x + w/2) - 1), y + i)
                        p_world_frame = self.img_to_3D(end_target_crodinates[0], end_target_crodinates[1])
                        i = i + 1
                    points_3D.append(p_world_frame[:3])
                    points.append(p_world_frame[:3])
                cv2.circle(cv_image, (end_target_crodinates[0], end_target_crodinates[1]), radius=5, color=(0, 0, 255), thickness=-1) # BGR color

                cloud = pc2.create_cloud_xyz32(header, points)
                navigation_cordinates.append(points_3D)
        pub.publish(cloud)
        self.result_pub.publish(self.bridge.cv2_to_imgmsg(cv_image))
        self.navigation_cordinates = navigation_cordinates


if __name__ == '__main__':
    try:
        image_processor = ImageProcessor()
        rate = rospy.Rate(10) # 10 Hz

        iterations = 0
        max_iterations = 50 # set this to your desired number of iterations

        while not rospy.is_shutdown() and iterations < max_iterations:
            # do any processing here
            
            rate.sleep()
            iterations += 1

        
        navigation_cordinates = image_processor.navigation_cordinates
        # print(navigation_cordinates)
        x = navigation_cordinates[0][0][0]
        y = navigation_cordinates[0][0][1]
        z = 0.2
        # print(navigation_cordinates)
        # print("========")
        # print(x, y, z)
        # move_robot(x, y, z)


        x2 = navigation_cordinates[0][1][0]
        y2 = navigation_cordinates[0][1][1]
        z2 = 0.2
        # move_robot(x2, y2, z2)

        start_point = [x, y]
        end_point = [x2, y2]
        start_point, end_point = np.array(start_point), np.array(end_point)
        points = np.linspace(start_point, end_point, 10)
        for i, point in enumerate(points):
            # print(f"Point {i}: {point}")
            move_robot(point[0], point[1], z)

    except rospy.ROSInterruptException:
        pass


