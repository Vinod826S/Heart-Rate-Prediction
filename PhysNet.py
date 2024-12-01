from __future__ import print_function, division
import gc
import torch
import pandas as pd
from torch import nn
import cv2
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Neg_Pearson(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson,self).__init__()
        return
    def forward(self, preds, labels):
        preds = preds.to(device) 
        labels = labels.to(device)      # tensor [Batch, Temporal]
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                # x
            sum_y = torch.sum(labels[i])               # y
            sum_xy = torch.sum(preds[i]*labels[i])        # xy
            sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))

            if (pearson>=0).data.cpu().numpy():    # torch.cuda.ByteTensor -->  numpy
               loss += 1 - pearson
            else:
               loss += 1 - torch.abs(pearson)            
            
        loss = loss/preds.shape[0]
        return loss
    
class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(self, frames = 128):  
        super(PhysNet_padding_Encoder_Decoder_MAX, self).__init__()
        
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x):	    	# x [3, T, 128,128]
        x_visual = x
        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)       # x [16, T, 64,64]
        
        x = self.ConvBlock2(x)		    # x [32, T, 64,64]
        x_visual6464 = self.ConvBlock3(x)	    	# x [32, T, 64,64]
        x = self.MaxpoolSpaTem(x_visual6464)      # x [32, T/2, 32,32]    Temporal halve
        
        x = self.ConvBlock4(x)		    # x [64, T/2, 32,32]
        x_visual3232 = self.ConvBlock5(x)	    	# x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x_visual3232)      # x [64, T/4, 16,16]
        

        x = self.ConvBlock6(x)		    # x [64, T/4, 16,16]
        x_visual1616 = self.ConvBlock7(x)	    	# x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x_visual1616)      # x [64, T/4, 8,8]

        x = self.ConvBlock8(x)		    # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)		    # x [64, T/4, 8, 8]
        x = self.upsample(x)		    # x [64, T/2, 8, 8]
        x = self.upsample2(x)		    # x [64, T, 8, 8]
        
        x = self.poolspa(x)     # x [64, T, 1,1]    -->  groundtruth left and right - 7 
        x = self.ConvBlock10(x)    # x [1, T, 1,1]
        
        rPPG = x.view(-1,length)            
        
        return rPPG

def safe_normalize(tensor):
    std = torch.std(tensor)
    return (tensor - torch.mean(tensor)) / (std + 1e-8)

def load_video_frames_2(video_path, num_frames=128, resize_shape=(128, 128)):
    frames = []
    detected_frame_count = 0
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    face_detected = False
    while not face_detected:
        ret, frame = cap.read()
        if not ret:
            raise ValueError("No face detected in the video.")  # Raise an error if all frames are exhausted
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            face_detected = True
            (x, y, w, h) = faces[0]  # Store the face region coordinates
            face_region = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_region, resize_shape)
            face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            frames.append(face_tensor)
            detected_frame_count += 1
    
    while detected_frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break 
        
        face_region = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_region, resize_shape)
        face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        frames.append(face_tensor)
        detected_frame_count += 1
    
    cap.release()
    
    if frames:
        frames_tensor = torch.cat(frames, dim=0)
        return frames_tensor, detected_frame_count
    else:
        raise ValueError("No faces detected in the video frames.")

class VideoDataset(Dataset):
    def __init__(self, video_paths, bvp_paths, resize_shape=(128, 128)):
        self.video_paths = video_paths
        self.bvp_paths = bvp_paths
        self.resize_shape = resize_shape
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        bvp_path = self.bvp_paths[idx]
        
        frames_tensor, detected_count = load_video_frames_2(video_path, resize_shape=self.resize_shape, num_frames=160)
        frames_tensor = frames_tensor.view(3, detected_count, *self.resize_shape)
        
        bvp_df = pd.read_csv(bvp_path)
        BVP_label = torch.tensor(bvp_df["BVP"].values, dtype=torch.float32).view(-1)
        BVP_label = safe_normalize(BVP_label)
        
        return frames_tensor, BVP_label
        
video_paths = []
bvp_paths = []

base_video_path = r"D:\UBFC\Training\Split_Training"

for l in range(1, 29):
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

# train_video_paths, val_video_paths, train_bvp_paths, val_bvp_paths = train_test_split(
#     video_paths, bvp_paths, test_size = 0.2, random_state=42
# )

train_dataset = VideoDataset(video_paths, bvp_paths)
# val_dataset = VideoDataset(val_video_paths, val_bvp_paths)

batch_size = 8
num_workers = 0
num_epochs = 15

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers = num_workers)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False, num_workers = num_workers)

model = PhysNet_padding_Encoder_Decoder_MAX(frames = 160).to(device)
optimizer = optim.Adam(model.parameters(), lr = 1e-4)

NegPea = Neg_Pearson().to(device)
model.train()

train_losses = []
# val_losses = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    epoch_loss = 0 

    for inputs, BVP_label in train_loader:
        inputs = inputs.to(device, dtype = torch.float32)
        BVP_label = BVP_label.to(device, dtype = torch.float32)
                    
        rPPG = model(inputs).to(device)

        rPPG = safe_normalize(rPPG)
        
        loss = NegPea(rPPG, BVP_label).to(device)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss for batch: {loss.item()}")

        del inputs, BVP_label, rPPG, loss 
        torch.cuda.empty_cache()
        gc.collect()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1} average training loss: {epoch_loss / len(train_loader)}")

    # model.eval()
    # val_loss = 0
    
    # with torch.no_grad():
    #     for inputs, BVP_label in val_loader:
    #         inputs = inputs.to(device, dtype = torch.float32)
    #         BVP_label = BVP_label.to(device, dtype = torch.float32)

    #         rPPG = model(inputs).to(device)

    #         rPPG = safe_normalize(rPPG)

    #         loss = NegPea(rPPG, BVP_label).to(device)
    #         val_loss += loss.item()

    #         print(f"Loss for batch: {loss.item()}")

    #         del inputs, BVP_label, rPPG, loss
    #         torch.cuda.empty_cache()
    #         gc.collect()

    # avg_val_loss = val_loss / len(val_loader)
    # val_losses.append(avg_val_loss)
    # print(f"Epoch {epoch + 1} average validation loss: {val_loss / len(val_loader)}")

model_save_path = r"D:\UBFC\Trained Models\PhysNet_LSTC.pt"
torch.save(model, model_save_path)

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
# plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# video_paths = []
# bvp_paths = []
# batch_size = 1

# base_video_path = r"D:\UBFC\Testing\Split_Testing_Phys"

# for l in range(29, 43):
#     k = 0
#     while True:
#         video_path = os.path.join(base_video_path, f"P-{l}", f"clip_{k+1}.avi")
#         bvp_path = os.path.join(base_video_path, f"P-{l}", f"clip_{k+1}_BVP.csv")
        
#         if os.path.isfile(video_path) and os.path.isfile(bvp_path):
#             video_paths.append(video_path)
#             bvp_paths.append(bvp_path)
#             k += 1
#         else:
#             break

# test_dataset = VideoDataset(video_paths, bvp_paths)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False, num_workers = num_workers)

# model_save_path = r"D:\UBFC\Trained Models\PhysNet.pt"
# model = torch.load(model_save_path)
# model.eval()

# for l in range(29, 43):
#     print(l)
#     for k in range(1, 17):
#         video_path = os.path.join(base_video_path, f"P-{l}", f"clip_{k}.avi")
#         bvp_path = os.path.join(base_video_path, f"P-{l}", f"clip_{k}_BVP.csv")
        
#         if os.path.isfile(video_path) and os.path.isfile(bvp_path):
#             test_dataset = VideoDataset([video_path], [bvp_path])
#             test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#             try:
#                 with torch.no_grad():
#                     for inputs, BVP_label in test_loader:
#                         inputs = inputs.to(device, dtype=torch.float32)
#                         BVP_label = BVP_label.to(device, dtype=torch.float32)
                                        
#                         rPPG = model(inputs).to(device)

#                         rPPG = safe_normalize(rPPG)
#                         rPPG_2 = rPPG.detach().cpu().numpy()
#                         df = pd.DataFrame({'Pulse': rPPG_2[0]})
#                         df.to_csv(rf"D:\UBFC\Testing\P-{l}\PhysNet\BVP_AVI\BVP_BatchSize8_{k}.csv", index=False)
                
#                         del inputs, rPPG, rPPG_2
#                         gc.collect()
#                         torch.cuda.empty_cache()
#             except (ValueError, OSError) as e:
#                 print(e)
#                 pass
