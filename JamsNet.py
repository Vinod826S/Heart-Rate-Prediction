from __future__ import print_function, division
import os
import gc
import torch
import pandas as pd
from torch import nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_normalize(tensor):
    std = torch.std(tensor)
    return (tensor - torch.mean(tensor)) / (std + 1e-8)

def cal_negative_pearson(x, y):
    " Negative Pearson loss function, x is the predicted value, y is the true value "
    x = x.to(device)
    y = y.to(device)

    n = len(x)
    sum_x = torch.sum(x)
    sum_y = torch.sum(y)
    sum_xy = torch.sum(torch.mul(x, y))
    sum_x2 = torch.sum(x.pow(2))
    sum_y2 = torch.sum(y.pow(2))
    molecular = n * sum_xy - torch.mul(sum_x, sum_y)
    denominator = torch.sqrt((sum_x2 * n - sum_x.pow(2))*(n * sum_y2 - sum_y.pow(2)))
    return (1 - molecular/denominator)

class CTJA (nn.Module):
    def __init__(self):
        super(CTJA, self).__init__()
        self.conv_D12 = nn.Conv3d(1, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1))
        self.conv_D13 = nn.Conv3d(1, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 2, 2), dilation=(1, 2, 2))
        self.conv_D14 = nn.Conv3d(1, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 4, 4), dilation=(1, 4, 4))
        self.bn_D11 = nn.BatchNorm3d(3)
        self.conv_D1D = nn.Conv3d(3, 3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, groups=3)
        self.bn_D12 = nn.BatchNorm3d(3)
        self.conv_D1P = nn.Conv3d(3, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.bn_D13 = nn.BatchNorm3d(1)

    def forward(self, x):
        T_C = x.size(1) * x.size(2)
        W_H = x.size(3) * x.size(4)
        x_0 = x.view(1, x.size(1), x.size(2), 1, W_H)
        x_0 = x_0.view(1, T_C, 1, 1, W_H)
        x_mean = torch.mean(x, 3, True)
        x_mean = torch.mean(x_mean, 4, True)
        x_mean = x_mean.permute(0, 4, 3, 1, 2)
        x_2 = self.conv_D12(x_mean)
        x_3 = self.conv_D13(x_mean)
        x_4 = self.conv_D14(x_mean)
        x_s = torch.stack([x_2, x_3, x_4], 0)
        x_s = x_s.squeeze(2).permute(1, 0, 2, 3, 4)
        x_s = self.bn_D11(x_s)
        x_s = F.elu(x_s)
        x_s = self.conv_D1D(x_s)
        x_s = self.conv_D1P(x_s)
        x_s = self.bn_D13(x_s)
        x_s = F.elu(x_s)
        x_sig = torch.sigmoid(x_s)
        x_sig = x_sig.view(1, 1, 1, T_C, 1)
        x_sig = x_sig.permute(0, 3, 4, 1, 2)
        x_sig = x_sig.repeat(1, 1, 1, 1, W_H)
        x_m = x_sig.mul(x_0)
        x_m = x_m.view(1, x.size(1), x.size(2), 1, W_H)
        x_m = x_m.view(1, x.size(1), x.size(2), x.size(3), x.size(4))
        
        return x_m

class STJA (nn.Module):
    def __init__(self):
        super(STJA, self).__init__()
        self.conv_st11 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn_st11 = nn.BatchNorm3d(1)
        self.conv_st12 = nn.Conv3d(1, 1, kernel_size=(1, 1, 1), stride=1, padding=0)
        self.bn_st12 = nn.BatchNorm3d(1)

    def forward(self, x):
        x_s = torch.mean(x, 1, True)
        x_s_1 = self.conv_st11(x_s)
        x_s_1 = self.bn_st11(x_s_1)
        x_s_1 = F.elu(x_s_1)
        x_t = torch.mean(torch.mean(x_s, 3, True), 4, True)
        x_t_1 = self.conv_st12(x_t)
        x_t_1 = self.bn_st12(x_t_1)
        x_t_1 = F.elu(x_t_1)
        x_s_sig = torch.sigmoid(x_s_1)
        x_t_sig = torch.sigmoid(x_t_1)
        x_t_sig = x_t_sig.repeat(1, 1, 1, x.size(3), x.size(4))
        x_st_sig = x_s_sig * x_t_sig
        x_st_sig = x_st_sig.repeat(1, x.size(1), 1, 1, 1)
        x = x * x_st_sig
        
        return x

class STJA_2D(nn.Module):
    def __init__(self):
        super(STJA_2D, self).__init__()

        self.conv_st11 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn_st11 = nn.BatchNorm2d(1)
        self.conv_st12 = nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn_st12 = nn.BatchNorm2d(1)

    def forward(self, x):
        x_s = torch.mean(x, 1, True)
        x_s_1 = self.conv_st11(x_s)
        x_s_1 = self.bn_st11(x_s_1)
        x_s_1 = F.elu(x_s_1)
        x_t = torch.mean(torch.mean(x_s, 2, True), 3, True)
        x_t_1 = self.conv_st12(x_t)
        x_t_1 = self.bn_st12(x_t_1)
        x_t_1 = F.elu(x_t_1)
        x_s_sig = torch.sigmoid(x_s_1)
        x_t_sig = torch.sigmoid(x_t_1)
        x_t_sig = x_t_sig.repeat(1, 1, x.size(2), x.size(3))
        x_st_sig = x_s_sig * x_t_sig
        x_st_sig = x_st_sig.repeat(1, x.size(1), 1, 1)
        x = x * x_st_sig
        
        return x

def gaussian_pyramid(video, num_levels=3):
        pyramid = []
        batch, length, channel, h, w = video.shape

        pyramid.append(video)

        for _ in range(num_levels -1):
            h = h//2
            w = w//2
            pyramid.append(torch.zeros(batch,length,channel, h, w).to(device))

        for i, frame in enumerate(video):
            for j in range(num_levels - 1):
                frame = TF.gaussian_blur(frame, kernel_size=3)
                frame = torch.nn.functional.interpolate(frame, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor = True)
                pyramid[j+1][i] = frame
        for i in range(len(pyramid)):
            pyramid[i] = pyramid[i]/255.
        return pyramid
    
class JAMSNet (nn.Module):
    def __init__(self):
        super(JAMSNet, self).__init__()
        adapt_h = 48
        adapt_w = 32
        T = 150
        # ##############  Multi-scale Feature Extraction & Fusion Net  ###############
        # ###   layer0   ###
        self.conv_00 = nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.bn_00 = nn.BatchNorm2d(32)
        self.adapt_avg_pool_01 = nn.AdaptiveAvgPool2d((adapt_h, adapt_w))
        self.conv_01 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_01 = nn.BatchNorm2d(32)
        self.conv_02 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_02 = nn.BatchNorm2d(32)
        # ####   layer1   ###
        self.conv_10 = nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.bn_10 = nn.BatchNorm2d(32)
        self.adapt_avg_pool_11 = nn.AdaptiveAvgPool2d((adapt_h, adapt_w))
        self.conv_11 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_11 = nn.BatchNorm2d(32)
        self.conv_12 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_12 = nn.BatchNorm2d(32)
        # ####   layer2   ###
        self.conv_20 = nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.bn_20 = nn.BatchNorm2d(32)
        self.adapt_avg_pool_21 = nn.AdaptiveAvgPool2d((adapt_h, adapt_w))
        self.conv_21 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_21 = nn.BatchNorm2d(32)
        self.conv_22 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_22 = nn.BatchNorm2d(32)

        # ##########################   Layer fuse   ##################################
        self.conv = nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0)
        self.adapt_avg_pool_L = nn.AdaptiveAvgPool2d((48, 32))

        # ##########################   rPPG Extraction Net   ##############################
        self.conv_1 = nn.Conv3d(32, 64, kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=0)
        self.bn_1 = nn.BatchNorm3d(64)
        # first
        self.max_pool_1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv_2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn_2 = nn.BatchNorm3d(64)
        self.conv_3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn_3 = nn.BatchNorm3d(64)
        # second
        self.max_pool_2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_4 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn_4 = nn.BatchNorm3d(64)
        self.conv_5 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn_5 = nn.BatchNorm3d(64)
        # third
        self.max_pool_3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv_6 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn_6 = nn.BatchNorm3d(64)
        self.conv_7 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn_7 = nn.BatchNorm3d(64)
        # fourth
        self.max_pool_4 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv_8 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn_8 = nn.BatchNorm3d(64)
        self.conv_9 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn_9 = nn.BatchNorm3d(64)
        # finally
        self.gobal_avg_pool3d = nn.AdaptiveAvgPool3d(output_size=(T, 1, 1))
        self.conv_L = nn.Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0)

        self.CTJA_1 = CTJA()
        self.CTJA_2 = CTJA()
        self.CTJA_3 = CTJA()
        self.CTJA_4 = CTJA()
        self.STJA_1 = STJA()
        self.STJA_2 = STJA()
        self.STJA_3 = STJA()
        self.STJA_4 = STJA()
        self.STJA_2D1 = STJA_2D()
        self.STJA_2D2 = STJA_2D()
        self.STJA_2D3 = STJA_2D()

    def forward(self, x0, x1, x2):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x0,x1,x2 ---> T C H W
        # ##############  Multi-scale Feature Extraction & Fusion Net  ###############
        # ###   layer0   ###
        x0 = self.conv_00(x0)
        x0 = self.bn_00(x0)
        x0 = F.elu(x0)
        x0 = self.adapt_avg_pool_01(x0)
        x0 = self.conv_01(x0)
        x0 = self.bn_01(x0)
        x0 = F.elu(x0)
        x0 = self.conv_02(x0)
        x0 = self.STJA_2D1(x0)
        x0 = self.bn_02(x0)
        x0 = F.elu(x0)
        # ####   layer1   ###
        x1 = self.conv_10(x1)
        x1 = self.bn_10(x1)
        x1 = F.elu(x1)
        x1 = self.adapt_avg_pool_11(x1)
        x1 = self.conv_11(x1)
        x1 = self.bn_11(x1)
        x1 = F.elu(x1)
        x1 = self.conv_12(x1)
        x1 = self.STJA_2D2(x1)
        x1 = self.bn_12(x1)
        x1 = F.elu(x1)
        # ####   layer2   ###
        x2 = self.conv_20(x2)
        x2 = self.bn_20(x2)
        x2 = F.elu(x2)
        x2 = self.adapt_avg_pool_21(x2)
        x2 = self.conv_21(x2)
        x2 = self.bn_21(x2)
        x2 = F.elu(x2)
        x2 = self.conv_22(x2)
        x2 = self.STJA_2D3(x2)
        x2 = self.bn_22(x2)
        x2 = F.elu(x2)
        
        # ##########################   Layer fuse   ##################################
        datemean_0 = torch.mean(x0)
        datemean_1 = torch.mean(x1)
        datemean_2 = torch.mean(x2)
        L = torch.zeros((1, 3))
        L[:, 0] = datemean_0
        L[:, 1] = datemean_1
        L[:, 2] = datemean_2
        L = L.reshape(1, 1, 3)
        L = L.to(device)
        L_conv = self.conv(L)
        L_conv = L_conv.squeeze(0)
        L_soft = torch.softmax(L_conv, 1)
        Data = x0 * L_soft[:, 0] + x1 * L_soft[:, 1] + x2 * L_soft[:, 2] 
        Data = self.adapt_avg_pool_L(Data)
        
        # ##########################   rPPG Extraction Net   #############################
        x = Data.unsqueeze(0) 
        x = x.permute(0, 2, 1, 3, 4)  
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = F.elu(x)
        # first 
        x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = self.CTJA_1(x)
        x = self.bn_2(x)
        x = F.elu(x)
        x = self.conv_3(x)
        x = self.STJA_1(x)
        x = self.bn_3(x)
        x = F.elu(x)
        # second
        x = self.max_pool_2(x)
        x = self.conv_4(x)
        x = self.CTJA_2(x)
        x = self.bn_4(x)
        x = F.elu(x)
        x = self.conv_5(x)
        x = self.STJA_2(x)
        x = self.bn_5(x)
        x = F.elu(x)
        # third
        x = self.max_pool_3(x)
        x = self.conv_6(x)
        x = self.CTJA_3(x)
        x = self.bn_6(x)
        x = F.elu(x)
        x = self.conv_7(x)
        x = self.STJA_3(x)
        x = self.bn_7(x)
        x = F.elu(x)
        # fourth
        x = self.max_pool_4(x)
        x = self.conv_8(x)
        x = self.CTJA_4(x)
        x = self.bn_8(x)
        x = F.elu(x)
        x = self.conv_9(x)
        x = self.STJA_4(x)
        x = self.bn_9(x)
        x = F.elu(x)
        # finally
        x = self.gobal_avg_pool3d(x)
        x = self.conv_L(x)
        x = x.squeeze(0).squeeze(0).squeeze(1)

        return x
    
def load_video_frames_2(video_path, num_frames=150, roi_size=(192, 128)):
    """
    Loads video frames, detects and tracks the face using KLT algorithm, and prepares sliding windows.

    Args:
        video_path (str): Path to the input video.
        num_frames (int): Number of frames in each sliding window.
        roi_size (tuple): Resize dimensions for the ROI (width, height).
        step_size (int): Step size for sliding windows.

    Returns:
        torch.Tensor: Tensor of processed frames divided into sliding windows.
    """
    frames = []

    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=5, blockSize=5)

    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    face_corners = None 
    previous_gray_frame = None
    roi_box = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if previous_gray_frame is None:
            face_corners = cv2.goodFeaturesToTrack(gray_frame, mask=None, **feature_params)
            if face_corners is None:
                raise ValueError("No features detected in the first frame.")

            x, y, w, h = cv2.boundingRect(face_corners)
            roi_box = (x, y, w, h)

        else:
            face_corners, _, _ = cv2.calcOpticalFlowPyrLK(previous_gray_frame, gray_frame, face_corners, None, **lk_params)
            if face_corners is None or len(face_corners) == 0:
                break

            x, y, w, h = cv2.boundingRect(face_corners)
            roi_box = (x, y, w, h)

        x, y, w, h = roi_box
        aspect_ratio = 3 / 2  
        if h / w > aspect_ratio:
            h = int(w * aspect_ratio)
        else:
            w = int(h / aspect_ratio)
        roi = frame[y:y + h, x:x + w]

        if roi is None or roi.size == 0:
            print("Empty ROI detected, skipping...")
            return None
        
        else:
            roi_resized = cv2.resize(roi, roi_size)

        face_tensor = torch.from_numpy(roi_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        frames.append(face_tensor)

        previous_gray_frame = gray_frame.copy()

        if len(frames) >= num_frames:
            break

    cap.release()

    if frames:
        frames_tensor = torch.cat(frames, dim=0)
        return frames_tensor
    else:
        raise ValueError("No faces detected in the video frames.")

    
class VideoDataset(Dataset):
    def __init__(self, video_paths, bvp_paths, resize_shape=(192, 128), num_frames=150):
        """
        VideoDataset for processing video frames and corresponding BVP labels.
        
        Args:
            video_paths (list): List of video file paths.
            bvp_paths (list): List of corresponding BVP file paths.
            resize_shape (tuple): Resize dimensions for the ROI (width, height).
            num_frames (int): Number of frames to process in each sliding window.
            step_size (int): Step size for sliding windows.
        """
        self.video_paths = video_paths
        self.bvp_paths = bvp_paths
        self.resize_shape = resize_shape
        self.num_frames = num_frames
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        bvp_path = self.bvp_paths[idx]
        
        frames_tensor = load_video_frames_2(
            video_path, 
            num_frames=self.num_frames, 
            roi_size=self.resize_shape 
        )
        
        frames_tensor = frames_tensor.view(-1, 3, *self.resize_shape)

        bvp_df = pd.read_csv(bvp_path)
        BVP_label = torch.tensor(bvp_df["BVP"].values, dtype=torch.float32).view(-1)

        BVP_label = safe_normalize(BVP_label)
        
        return frames_tensor, BVP_label
    
# video_paths = []
# bvp_paths = []

# base_video_path = r"D:\UBFC\Training\Split_Training_JAMS"

# for l in range(1, 36):
#     k = 0
#     while True:
#         video_path = os.path.join(base_video_path, f"P-{l}", f"window_{k+1}.avi")
#         bvp_path = os.path.join(base_video_path, f"P-{l}", f"window_{k+1}_BVP.csv")
        
#         if os.path.isfile(video_path) and os.path.isfile(bvp_path):
#             video_paths.append(video_path)
#             bvp_paths.append(bvp_path)
#             k += 1
#         else:
#             break

# train_video_paths, val_video_paths, train_bvp_paths, val_bvp_paths = train_test_split(
#     video_paths, bvp_paths, test_size = 0.2, random_state=42
# )

# train_dataset = VideoDataset(video_paths, bvp_paths)
# val_dataset = VideoDataset(val_video_paths, val_bvp_paths)

batch_size = 1
num_workers = 0
# num_epochs = 10

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers = num_workers)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False, num_workers = num_workers)

# model = JAMSNet().to(device)
# optimizer = optim.Adam(model.parameters(), lr = 1e-4)

# train_losses = []
# val_losses = []

# for epoch in range(num_epochs):
#     print(f"Epoch {epoch + 1}/{num_epochs}")
#     model.train()
#     epoch_loss = 0 

#     for inputs, BVP_label in train_loader:
#         inputs = inputs.to(device, dtype = torch.float32)
#         BVP_label = BVP_label.to(device, dtype = torch.float32)
#         BVP_label = BVP_label.T
        
#         py = gaussian_pyramid(inputs)
#         py = [p.squeeze(0) for p in py]
                    
#         rPPG = model(py[0], py[1], py[2]).to(device)

#         rPPG = safe_normalize(rPPG)
        
#         loss = cal_negative_pearson(rPPG, BVP_label)
#         epoch_loss += loss.item()

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         print(f"Loss for batch: {loss.item()}")

#         del inputs, BVP_label, rPPG, loss 
#         torch.cuda.empty_cache()
#         gc.collect()

#     avg_train_loss = epoch_loss / len(train_loader)
#     train_losses.append(avg_train_loss)
#     print(f"Epoch {epoch + 1} average training loss: {epoch_loss / len(train_loader)}")

#     model.eval()
#     val_loss = 0
    
#     with torch.no_grad():
#         for inputs, BVP_label in val_loader:
#             inputs = inputs.to(device, torch.float32)
#             BVP_label = BVP_label.to(device, torch.float32)
#             BVP_label = BVP_label.T

#             py = gaussian_pyramid(inputs)
#             py = [p.squeeze(0) for p in py]

#             rPPG = model(py[0], py[1], py[2]).to(device)

#             rPPG = safe_normalize(rPPG)

#             loss = cal_negative_pearson(rPPG, BVP_label)
#             val_loss += loss.item()

#             print(f"Loss for batch: {loss.item()}")

#             del inputs, BVP_label, rPPG, loss
#             torch.cuda.empty_cache()
#             gc.collect()

#     avg_val_loss = val_loss / len(val_loader)
#     val_losses.append(avg_val_loss)
#     print(f"Epoch {epoch + 1} average validation loss: {val_loss / len(val_loader)}")

# for epoch in range(num_epochs):
#     print(f"Epoch {epoch + 1}/{num_epochs}")
#     model.train()
#     epoch_loss = 0

#     optimizer.zero_grad()

#     for i, (inputs, BVP_label) in enumerate(train_loader):
#         inputs = inputs.to(device, dtype=torch.float32)
#         BVP_label = BVP_label.to(device, dtype=torch.float32)

#         py = gaussian_pyramid(inputs)
#         py = [p.squeeze(0) for p in py]

#         rPPG = model(py[0], py[1], py[2]).to(device)
#         rPPG = safe_normalize(rPPG.unsqueeze(0))

#         loss = cal_negative_pearson(rPPG, BVP_label)
#         loss = loss / accumulation_steps
#         loss.backward()

#         if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
#             optimizer.step()
#             optimizer.zero_grad()

#         epoch_loss += loss.item()

#         del inputs, BVP_label, rPPG, loss
#         torch.cuda.empty_cache()
#         gc.collect()

#     avg_train_loss = epoch_loss / len(train_loader)
#     train_losses.append(avg_train_loss)
#     print(f"Epoch {epoch + 1} average training loss: {avg_train_loss}")

#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for inputs, BVP_label in val_loader:
#             inputs = inputs.to(device, dtype=torch.float32)
#             BVP_label = BVP_label.to(device, dtype=torch.float32)

#             py = gaussian_pyramid(inputs)
#             py = [p.squeeze(0) for p in py]

#             rPPG = model(py[0], py[1], py[2]).to(device)
#             rPPG = safe_normalize(rPPG.unsqueeze(0))

#             loss = cal_negative_pearson(rPPG, BVP_label)
#             val_loss += loss.item()

#             del inputs, BVP_label, rPPG, loss
#             torch.cuda.empty_cache()
#             gc.collect()

#     avg_val_loss = val_loss / len(val_loader)
#     val_losses.append(avg_val_loss)
#     print(f"Epoch {epoch + 1} average validation loss: {avg_val_loss}")

#     scheduler.step(avg_val_loss)  

# model_save_path = r"D:\UBFC\Trained Models\JAMSNet_3.pt"
# torch.save(model, model_save_path)

# plt.figure(figsize=(10, 5))
# plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
# plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o')
# plt.title('Training vs Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid()
# plt.show()

video_paths = []
bvp_paths = []
batch_size = 1

base_video_path = r"D:\UBFC\Testing\Split_Testing_JAMS"

for l in range(36, 43):
    k = 0
    while True:
        video_path = os.path.join(base_video_path, f"P-{l}", f"clip_{k+1}.avi")
        bvp_path = os.path.join(base_video_path, f"P-{l}", f"clip_{k+1}_BVP.csv")
        
        if os.path.isfile(video_path) and os.path.isfile(bvp_path):
            video_paths.append(video_path)
            bvp_paths.append(bvp_path)
            k += 1
        else:
            break

test_dataset = VideoDataset(video_paths, bvp_paths)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False, num_workers = num_workers)

model_save_path = r"D:\UBFC\Trained Models\JAMSNet_3.pt"
model = torch.load(model_save_path)
model.eval()

for l in range(40, 43):
    print(l)
    for k in range(1, 69):
        try:
            video_path = os.path.join(base_video_path, f"P-{l}", f"clip_{k}.avi")
            bvp_path = os.path.join(base_video_path, f"P-{l}", f"clip_{k}_BVP.csv")
        
            if os.path.isfile(video_path) and os.path.isfile(bvp_path):
                test_dataset = VideoDataset([video_path], [bvp_path])
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


                with torch.no_grad():
                    for inputs, BVP_label in test_loader:
                        inputs = inputs.to(device, dtype=torch.float32)
                        BVP_label = BVP_label.to(device, dtype=torch.float32)
                                        
                        py = gaussian_pyramid(inputs)
                        py = [p.squeeze(0) for p in py]

                        rPPG = model(py[0], py[1], py[2]).to(device)

                        rPPG = safe_normalize(rPPG)
                        rPPG_2 = rPPG.detach().cpu().numpy()
                        df = pd.DataFrame({'Pulse': rPPG_2[:, 0]})
                        df.to_csv(rf"D:\UBFC\Testing\P-{l}\JAMSNet\BVP_AVI\BVP_3_{k}.csv", index=False)
                
                        del inputs, rPPG, rPPG_2
                        gc.collect()
                        torch.cuda.empty_cache()
        except (ValueError, OSError) as e:
            print(e)
            pass


