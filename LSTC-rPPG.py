from __future__ import print_function, division
import os
import gc
import torch
import pandas as pd
from torch import nn
import cv2
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.utils import _triple
from torch.utils.data import Dataset, DataLoader
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_normalize(tensor):
    std = torch.std(tensor)
    return (tensor - torch.mean(tensor)) / (std + 1e-8)

class HybridLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, lambda_val=0.2):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_val = lambda_val
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted, target):
        eps = 1e-8
        time_loss = self.mse_loss(predicted, target)
        
        pred_psd = torch.fft.fft(predicted, dim=-1).abs() + eps
        target_psd = torch.fft.fft(target, dim=-1).abs() + eps
        di = torch.log(pred_psd) - torch.log(target_psd)
        freq_loss = (di ** 2).mean() - (self.lambda_val / (di.numel() ** 2)) * (di.sum() ** 2)
        
        total_loss = self.alpha * time_loss + self.beta * freq_loss
        return total_loss

class LSTCrPPG(torch.nn.Module):
    def __init__(self, frames = 160):
        super(LSTCrPPG, self).__init__()
        self.encoder_block = EncoderBlock()
        self.decoder_block = DecoderBlock()

    def forward(self, x):
        e = self.encoder_block(x)
        out = self.decoder_block(e)
        return out.squeeze()


class EncoderBlock(torch.nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()
        self.encoder_block1 = torch.nn.Sequential(
            ConvBlock3D(3, 16, [3,3,3], [1,1,1], [1,1,1]),
            ConvBlock3D(16, 16, [3,3,3], [1,1,1], [1,1,1]),
            torch.nn.BatchNorm3d(16)
        )
        self.encoder_block2 = torch.nn.Sequential(
            torch.nn.AvgPool3d(2),
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [1,1,1]),
            torch.nn.BatchNorm3d(16)
        )
        self.encoder_block3 = torch.nn.Sequential(
            torch.nn.AvgPool3d(2),
            ConvBlock3D(16, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            torch.nn.BatchNorm3d(32)
        )
        self.encoder_block4 = torch.nn.Sequential(
            torch.nn.AvgPool3d(2),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            torch.nn.BatchNorm3d(32)
        )
        self.encoder_block5 = torch.nn.Sequential(
            torch.nn.AvgPool3d(2),
            ConvBlock3D(32, 64, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1,1,1]),
            torch.nn.BatchNorm3d(64)
        )
        self.encoder_block6 = torch.nn.Sequential(
            torch.nn.AvgPool3d(2),
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1,1,1]),
            torch.nn.BatchNorm3d(64)
        )
        self.encoder_block7 = torch.nn.Sequential(
            ConvBlock3D(64, 64, [5, 3, 3], [1, 1, 1], [0,1,1]),
            torch.nn.BatchNorm3d(64)
        )

    def forward(self, x):
        e1 = self.encoder_block1(x)
        e2 = self.encoder_block2(e1)
        e3 = self.encoder_block3(e2)
        e4 = self.encoder_block4(e3)
        e5 = self.encoder_block5(e4)
        e6 = self.encoder_block6(e5)
        e7 = self.encoder_block7(e6)

        return [e7,e6,e5,e4,e3,e2,e1]

class DecoderBlock(torch.nn.Module):
    def __init__(self):
        super(DecoderBlock, self).__init__()
        self.decoder_block6_transpose = torch.nn.ConvTranspose3d(64,64,[5,1,1],[1,1,1])
        self.decoder_block6 = torch.nn.Sequential(
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1,1,1]),
            torch.nn.BatchNorm3d(64)
        )
        self.decoder_block5_transpose =torch.nn.ConvTranspose3d(64, 64, [4, 1, 1],[2,1,1])
        self.decoder_block5 = torch.nn.Sequential(
            ConvBlock3D(64, 32, [3, 3, 3], [1, 1, 1],[1,1,1]),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1],[0,1,1]),
            torch.nn.BatchNorm3d(32)
        )
        self.decoder_block4_transpose = torch.nn.ConvTranspose3d(32, 32, [4, 1, 1],[2,1,1])
        self.decoder_block4 = torch.nn.Sequential(
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [0,1,1]),
            torch.nn.BatchNorm3d(32)
        )
        self.decoder_block3_transpose = torch.nn.ConvTranspose3d(32, 32, [4, 1, 1],[2,1,1])
        self.decoder_block3 = torch.nn.Sequential(
            ConvBlock3D(32, 16, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [0,1,1]),
            torch.nn.BatchNorm3d(16)
        )
        self.decoder_block2_transpose = torch.nn.ConvTranspose3d(16, 16, [4, 1, 1],[2,1,1])
        self.decoder_block2 = torch.nn.Sequential(
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [0,1,1]),
            torch.nn.BatchNorm3d(16)
        )
        self.decoder_block1_transpose = torch.nn.ConvTranspose3d(16, 16, [4, 1, 1],[2,1,1])
        self.decoder_block1 = torch.nn.Sequential(
            ConvBlock3D(16, 3, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(3, 3, [3, 3, 3], [1, 1, 1], [0,1,1]),
            torch.nn.BatchNorm3d(3)
        )
        self.predictor = torch.nn.Conv3d(3, 1 ,[1,4,4])

    def forward(self, encoded_feature):
        d6 = self.decoder_block6(self.TARM(encoded_feature[1], self.decoder_block6_transpose(encoded_feature[0])))
        d5 = self.decoder_block5(self.TARM(encoded_feature[2],self.decoder_block5_transpose(d6)))
        d4 = self.decoder_block4(self.TARM(encoded_feature[3],self.decoder_block4_transpose(d5)))
        d3 = self.decoder_block3(self.TARM(encoded_feature[4],self.decoder_block3_transpose(d4)))
        d2 = self.decoder_block2(self.TARM(encoded_feature[5],self.decoder_block2_transpose(d3)))
        d1 = self.decoder_block1(self.TARM(encoded_feature[6],self.decoder_block1_transpose(d2)))
        predictor = self.predictor(d1)
        return predictor

    def TARM(self, e,d):
        target = d
        shape = d.shape
        e = torch.nn.functional.adaptive_avg_pool3d(e,d.shape[2:])
        e = e.view(e.shape[0],e.shape[1], shape[2],-1)
        d = d.view(d.shape[0], shape[1], shape[2], -1)
        temporal_attention_map = e @ torch.transpose(d,3,2)
        temporal_attention_map = torch.nn.functional.softmax(temporal_attention_map,dim=-1)
        refined_map = temporal_attention_map@e
        out = ( 1 + torch.reshape(refined_map,shape)) * target
        return out

class ConvBlock3D(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = torch.nn.Sequential(
            torch.nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding),
            torch.nn.ELU()
        )

    def forward(self, x):
        return self.conv_block_3d(x)

def load_video_frames_2(video_path, num_frames = 160, resize_shape=(128, 128)):
    frames = []
    detected_frame_count = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")
    
    mtcnn = MTCNN()
    face_box = None 
    
    while detected_frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if detected_frame_count == 0:
            detections = mtcnn.detect_faces(frame)
            if detections:
                x, y, w, h = detections[0]['box']
                
                center_x, center_y = x + w // 2, y + h // 2
                w, h = int(w * 1.6), int(h * 1.6)
                x, y = max(0, center_x - w // 2), max(0, center_y - h // 2)
                
                face_box = (x, y, w, h)
            else:
                raise ValueError("No face detected in the first frame.")
        
        if face_box:
            x, y, w, h = face_box
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
        
        frames_tensor, detected_count = load_video_frames_2(video_path, resize_shape=self.resize_shape, num_frames = 160)
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

train_video_paths, val_video_paths, train_bvp_paths, val_bvp_paths = train_test_split(
    video_paths, bvp_paths, test_size = 0.2, random_state=42
)

train_dataset = VideoDataset(video_paths, bvp_paths)
val_dataset = VideoDataset(val_video_paths, val_bvp_paths)

batch_size = 1
num_workers = 0
num_epochs = 10

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers = num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False, num_workers = num_workers)

model = LSTCrPPG().to(device)
optimizer = optim.Adam(model.parameters(), lr = 5e-5).to(device)
siloss = HybridLoss(alpha = 1.0, beta = 0.5, lambda_val = 0.2).to(device)
model.train()

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    epoch_loss = 0 

    for inputs, BVP_label in train_loader:
        print(epoch + 1)
        inputs = inputs.to(device, dtype = torch.float32)
        BVP_label = BVP_label.to(device, dtype = torch.float32)
                    
        rPPG = model(inputs).to(device)
        rPPG = rPPG.unsqueeze(0)

        rPPG = safe_normalize(rPPG)
        
        loss = siloss(rPPG, BVP_label).to(device)
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

    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for inputs, BVP_label in val_loader:
            inputs = inputs.to(device, dtype = torch.float32)
            BVP_label = BVP_label.to(device, dtype = torch.float32)

            rPPG = model(inputs).to(device)
            rPPG = rPPG.unsqueeze(0)

            rPPG = safe_normalize(rPPG)

            loss = siloss(rPPG, BVP_label).to(device)
            val_loss += loss.item()

            print(f"Loss for batch: {loss.item()}")

            del inputs, BVP_label, rPPG, loss
            torch.cuda.empty_cache()
            gc.collect()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch + 1} average validation loss: {val_loss / len(val_loader)}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

model_save_path = r"D:\UBFC\Trained Models\LSTC-rPPG_4.pt"
torch.save(model, model_save_path)

# base_video_path_2 = r"D:\UBFC\Testing\Split_Testing"

# model = torch.load(model_save_path)
# model.eval()

# for l in range(29, 43):
#     print(l)
#     for k in range(1, 13):
#         video_path = os.path.join(base_video_path_2, f"P-{l}", f"clip_{k}.avi")
#         bvp_path = os.path.join(base_video_path_2, f"P-{l}", f"clip_{k}_BVP.csv")
        
#         if os.path.isfile(video_path) and os.path.isfile(bvp_path):
#             test_dataset = VideoDataset([video_path], [bvp_path])
#             test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#             try:
#                 for inputs, BVP_label in test_loader:
#                     inputs = inputs.to(device, dtype=torch.float32)
#                     BVP_label = BVP_label.to(device, dtype=torch.float32)
                                        
#                     with torch.no_grad():
#                         rPPG = model(inputs)

#                     rPPG = safe_normalize(rPPG)
#                     rPPG_2 = rPPG.detach().cpu().numpy()
#                     df = pd.DataFrame({'Pulse': rPPG_2})
#                     df.to_csv(rf"D:\UBFC\Testing\P-{l}\LSTC-rPPG\BVP_AVI\BVP_Trained_Clip_2_{k}.csv", index=False)
            
#                     del inputs, rPPG, rPPG_2
#                     gc.collect()
#                     torch.cuda.empty_cache()
#             except (ValueError, OSError) as e:
#                 print(e)
#                 pass


