import os
import os.path as osp
import random
import numpy as np
from torchvision.datasets.folder import make_dataset
# np.set_num_threads(1)
from decord import VideoReader
import glob
from tqdm import tqdm
import pickle
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from decord import cpu, gpu
from torchvision.io import read_video, write_video
import pandas as pd
import cv2
# cv2.setNumThreads(0)  # 禁用OpenCV多线程
# cv2.ocl.setUseOpenCL(False)  # 禁用OpenCL加速（部分情况下有效）

import json
import traceback
from natsort import natsorted
from typing import Tuple, List

class AMDConsecutiveVideo(Dataset):
    def __init__(
            self, 
            video_dir: str      = '', # video dir or pkl file
            sample_size: int    = 32, 
            sample_stride: int  = 2, 
            sample_n_frames:int = 16,
            ref_drop_ratio = 0.0,
            data_frac = 1.0,
            target_fps = 8,
            use_grey = True,
            use_mask = False,
            mask_video_ratio = 0,
        ):
        
        # Init setting
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.ref_drop_ratio = ref_drop_ratio
        self.data_frac = data_frac # 是否以一定概率drop一部分数据
        self.use_grey = use_grey
        self.use_mask = use_mask
        self.mask_video_ratio = mask_video_ratio
        self.target_fps = target_fps

        # Transform
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size) # (256,256)
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        if "csv" in video_dir:
            metadata = pd.read_csv(
                video_dir,
                on_bad_lines="skip",
                encoding="ISO-8859-1",
                engine="python",
                sep=",",)

            if self.data_frac < 1:
                metadata = metadata.sample(frac=self.data_frac)
            metadata.dropna(inplace=True) # 使用 dropna(inplace=True) 方法删除所有包含缺失值（NaN）的行

            # Data dict
            self.metadata_list = []
            for file_path in tqdm(metadata['videos']):
                d = {}
                d['name'] = self.get_file_name(file_path)
                d['video_path'] = file_path
                self.metadata_list.append(d)

        else:
            if 'pkl' in video_dir:
                with open(video_dir, 'rb') as f:
                    video_files = pickle.load(f)
                print(f'Total {len(video_files)} !!!')
            elif '.txt' in video_dir:
                with open(video_dir, 'r') as file:
                    lines = file.readlines()
                video_dirs = [line.strip() for line in lines]

                video_files = []
                for dir in video_dirs:
                    video_files += glob.glob(os.path.join(dir, '**', '*.mp4'), recursive=True)
                print(f'Total {len(video_files)} !!!')
        
            else:
                video_files = glob.glob(os.path.join(video_dir, '**', '*.mp4'), recursive=True)

            # Data dict
            self.metadata_list = []
            for file_path in tqdm(video_files):
                d = {}
                d['name'] = self.get_file_name(file_path)
                d['video_path'] = file_path
                self.metadata_list.append(d)


        self.length = len(self.metadata_list)
        print(f'Total {self.length} files is available')
    
    def __getitem__(self, idx):
        while True:
            try:
                file_name = self.metadata_list[idx]['name']
                if self.use_grey:
                    if self.use_mask:
                        file_name,videos,ref_img,ref_grey_img,grey_videos,camera_mask = self.get_batch(idx)
                    else:
                        file_name,videos,ref_img,ref_grey_img,grey_videos = self.get_batch(idx)
                else:
                    file_name,videos,ref_img = self.get_batch(idx)
                # file_name,videos,ref_img = self.get_batch(idx)
                break

            except Exception as e:
                # file_name = self.metadata_list[idx]['name']
                # print(file_name)
                print('error',e)
                idx = random.randint(0, self.length-1)
        if self.use_grey:
            if self.use_mask:
                sample = {
                    "ref_grey_img":ref_grey_img,
                    "grey_videos":grey_videos,
                    "name":file_name,
                    "videos":videos,
                    "ref_img":ref_img,
                    "camera_mask":camera_mask,
                }
            else:
                sample = {
                    "ref_grey_img":ref_grey_img,
                    "grey_videos":grey_videos,
                    "name":file_name,
                    "videos":videos,
                    "ref_img":ref_img,
                }
        else:
            sample = {
                "name":file_name,
                "videos":videos,
                "ref_img":ref_img,
            }
        # sample = dict(name=file_name,videos=videos,ref_img = ref_img)
        return sample 

    def __len__(self):
        return self.length

    def get_batch(self, idx):
        
        # init
        meta_data = self.metadata_list[idx]
        file_name = meta_data['name']
        video_path = meta_data['video_path']

        # video process
        video_reader = VideoReader(video_path, ctx=cpu(0))
        video_length = len(video_reader)
        video_fps = video_reader.get_avg_fps()
        # print(video_fps,"_____________video_fps________")
        
        # sample_frames = self.sample_n_frames + 1 # refimg + videos
        # clip_length = min(video_length, (sample_frames - 1) * self.sample_stride + 1)
        # start_idx   = random.randint(0, video_length)
        # batch_index = np.linspace(start_idx, start_idx + clip_length - 1, sample_frames, dtype=int)

        batch_index = self.sample_frames_with_fps(video_length,video_fps,self.sample_n_frames+1,self.target_fps)

        if len(batch_index) != self.sample_n_frames+1:
            raise ValueError("The number of sampled frames does not match the expected count.")

        if self.use_grey:

            frames = video_reader.get_batch(batch_index).asnumpy()
            grey_frames = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]))
            for i in range(frames.shape[0]):
                # 将 RGB 转换为 BGR，因为 OpenCV 默认使用 BGR 格式
                bgr_frame = cv2.cvtColor(frames[i, ...], cv2.COLOR_RGB2BGR)
                # 将 BGR 转换为灰度
                grey_frames[i, ...] = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            grey_videos = torch.from_numpy(grey_frames).unsqueeze(1).contiguous()
            grey_videos = grey_videos.repeat(1,3,1,1)
            grey_videos = grey_videos / 255.0
            grey_videos_cache = self.pixel_transforms(grey_videos).to(dtype=torch.float32)
            grey_videos = grey_videos_cache[1:,:,:,:] # F,C,H,W
            grey_ref_img = grey_videos_cache[0,:,:,:] # C,H,W
            # grey_ref_img = grey_videos_cache[1,:,:,:] # C,H,W
            grey_ref_img = grey_ref_img.unsqueeze(0).repeat(grey_videos.shape[0],1,1,1)

            if self.use_mask:
                frame1 = frames[0]
                frame2 = frames[-1]
                camera_mask, object_mask = flow_mask(frame1, frame2,mask_video_ratio=self.mask_video_ratio)
                camera_mask = torch.tensor(camera_mask).to(dtype=torch.float32)

                camera_mask = camera_mask.unsqueeze(0).unsqueeze(0)
                camera_mask = camera_mask.repeat(self.sample_n_frames*2,4,1,1)


        videos = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous() #(N,H,W,C)->(N,C,H,W)
        videos = videos / 255.0 

        # # ref frame
        # idx_all = np.arange(0,video_length)
        # occ_idx = np.arange(start_idx, start_idx + clip_length)
        # ref_idx = [x for x in idx_all if x not in occ_idx]
        # if len(ref_idx) == 0:
        #     ref_frame_idx = batch_index[0]
        # else:
        #     np.random.shuffle(ref_idx)
        #     ref_frame_idx = ref_idx[0]
        # ref_frame = torch.from_numpy(video_reader[ref_frame_idx].asnumpy()).permute(2, 0, 1).contiguous() 
        # ref_frame = ref_frame / 255.0
        
        # transform
        videos_cache = self.pixel_transforms(videos) # F+1,C,H,W
        videos = videos_cache[1:,:,:,:] # F,C,H,W
        ref_img = videos_cache[0,:,:,:] # C,H,W
        # ref_img = videos_cache[1,:,:,:]

        # repeat
        ref_img = ref_img.unsqueeze(0).repeat(videos.shape[0],1,1,1) # F,C,H,W

        if self.use_grey:
            if self.use_mask:
                return file_name,videos,ref_img,grey_ref_img,grey_videos,camera_mask
            else:
                return file_name,videos,ref_img,grey_ref_img,grey_videos
        else:
            return file_name,videos,ref_img
        # return file_name,videos,ref_frame

    def sample_frames_with_fps(
        self,
        total_frames,
        video_fps,
        sample_num_frames,
        sample_fps,
        start_index=None
    ):
        """sample frames proportional to the length of the frames in one second
        e.g., 1s video has 30 frames, when 'fps'=3, we sample frames with spacing of 30/3=10
        return the frame indices

        Parameters
        ----------
        total_frames : length of the video
        video_fps : original fps of the video
        sample_num_frames : number of frames to sample
        sample_fps : the fps to sample frames
        start_index : the starting frame index. If it is not None, it will be used as the starting frame index  

        Returns
        -------
        frame indices
        """
        # sample_num_frames = min(sample_num_frames, total_frames)
        interval = round(video_fps / sample_fps)
        frames_range = (sample_num_frames - 1) * interval + 1

        if start_index is not None:
            start = start_index
        elif total_frames - frames_range - 1 < 0:
            start = 0
        else:
            start = random.randint(0, total_frames - frames_range - 1)

        frame_idxs = np.linspace(
            start=start, stop=min(total_frames - 1, start + frames_range), num=sample_num_frames
        ).astype(int)

        return frame_idxs

    # frame_idxs = sample_frames_with_fps(
    #     total_frames=total_frames,
    #     video_fps=video_fps,
    #     sample_num_frames=sample_num_frames,
    #     sample_fps=sample_fps,
    #     start_index=start_index
    # )

    def get_file_name(self, file_path):
        return file_path.split('/')[-1].split('.')[0]
    
    @staticmethod
    def collate_fn(batch):
        # name
        name = [item['name'] for item in batch]

        # videos
        videos = [item['videos'] for item in batch]
        videos = torch.stack(videos)

        # ref_img
        ref_img = [item['ref_img'] for item in batch]
        ref_img = torch.stack(ref_img)

        # # videos
        grey_videos = [item['grey_videos'] for item in batch]
        grey_videos = torch.stack(grey_videos)

        # # ref_img
        ref_grey_img = [item['ref_grey_img'] for item in batch]
        ref_grey_img = torch.stack(ref_grey_img)

        return dict(name=name, videos=videos, ref_img=ref_img, grey_videos=grey_videos, ref_grey_img=ref_grey_img)

        # return dict(name=name, videos=videos, ref_img=ref_img)

class AMDRandomPair(Dataset):
    def __init__(
            self, 
            video_dir: str      = '', # video dir or pkl file
            sample_size: int    = 32, 
            sample_stride: int  = 4, 
            sample_n_frames:int = 16,
            ref_drop_ratio = 0.0,
            use_grey = True,
            use_mask = True,
            mask_video_ratio = 0.6,
            target_fps = 8,
        ):
        
        # Init setting
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.ref_drop_ratio = ref_drop_ratio
        self.use_grey = use_grey
        self.use_mask = use_mask
        self.mask_video_ratio = mask_video_ratio
        
        # Transform
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size) # (256,256)
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(min(sample_size)),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        if "csv" in video_dir:
            metadata = pd.read_csv(
                video_dir,
                on_bad_lines="skip",
                encoding="ISO-8859-1",
                engine="python",
                sep=",",)

            metadata.dropna(inplace=True) # 使用 dropna(inplace=True) 方法删除所有包含缺失值（NaN）的行

            # Data dict
            self.metadata_list = []
            for file_path in tqdm(metadata['videos']):
                d = {}
                d['name'] = self.get_file_name(file_path)
                d['video_path'] = file_path
                self.metadata_list.append(d)

        else:
            if 'pkl' in video_dir:
                with open(video_dir, 'rb') as f:
                    video_files = pickle.load(f)
                print(f'Total {len(video_files)} !!!')
            elif '.txt' in video_dir:
                with open(video_dir, 'r') as file:
                    lines = file.readlines()
                video_dirs = [line.strip() for line in lines]

                video_files = []
                for dir in video_dirs:
                    video_files += glob.glob(os.path.join(dir, '**', '*.mp4'), recursive=True)
                print(f'Total {len(video_files)} !!!')
        
            else:
                video_files = glob.glob(os.path.join(video_dir, '**', '*.mp4'), recursive=True)

            # Data dict
            self.metadata_list = []
            for file_path in tqdm(video_files):
                d = {}
                d['name'] = self.get_file_name(file_path)
                d['video_path'] = file_path
                self.metadata_list.append(d)
                
        self.length = len(self.metadata_list)
        print(f'Total {self.length} files is available')
    
    def __getitem__(self, idx):
        while True:
            try:
                file_name = self.metadata_list[idx]['name']
                if self.use_grey:
                    if self.use_mask:
                        file_name,videos,ref_img,ref_grey_img,grey_videos,camera_mask = self.get_batch(idx)
                    else:
                        file_name,videos,ref_img,ref_grey_img,grey_videos = self.get_batch(idx)
                else:
                    file_name,videos,ref_img = self.get_batch(idx)
                break

            except Exception as e:
                # file_name = self.metadata_list[idx]['name']
                # print(file_name)
                print('error',e)
                idx = random.randint(0, self.length-1)

        # file_name = self.metadata_list[idx]['name']
        # if self.use_grey:
        #     file_name,videos,ref_img,ref_grey_img,grey_videos = self.get_batch(idx)
        # else:
        #     file_name,videos,ref_img = self.get_batch(idx)
        
        if self.use_grey:
            if self.use_mask:
                sample = {
                    "ref_grey_img":ref_grey_img,
                    "grey_videos":grey_videos,
                    "name":file_name,
                    "videos":videos,
                    "ref_img":ref_img,
                    "camera_mask":camera_mask,
                }
            else:
                sample = {
                    "ref_grey_img":ref_grey_img,
                    "grey_videos":grey_videos,
                    "name":file_name,
                    "videos":videos,
                    "ref_img":ref_img,
                }
        else:
            sample = {
                "name":file_name,
                "videos":videos,
                "ref_img":ref_img,
            }
        return sample 

    def __len__(self):
        return self.length

    def get_batch(self, idx):
        # init
        meta_data = self.metadata_list[idx]
        file_name = meta_data['name']
        video_path = meta_data['video_path']

        # video process
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)
        

        ref_idx,video_idx = generate_non_equal_random_lists(frame_num=video_length,sample_num=self.sample_n_frames)

        if self.use_grey:
            ref_frames = video_reader.get_batch(ref_idx).asnumpy()
            ref_grey_frames = np.zeros((ref_frames.shape[0], ref_frames.shape[1], ref_frames.shape[2]))
            for i in range(ref_frames.shape[0]):
                # 将 RGB 转换为 BGR，因为 OpenCV 默认使用 BGR 格式
                ref_bgr_frame = cv2.cvtColor(ref_frames[i, ...], cv2.COLOR_RGB2BGR)
                # 将 BGR 转换为灰度
                ref_grey_frames[i, ...] = cv2.cvtColor(ref_bgr_frame, cv2.COLOR_BGR2GRAY)
            ref_grey_videos = torch.from_numpy(ref_grey_frames).unsqueeze(1).contiguous()
            ref_grey_videos = ref_grey_videos.repeat(1,3,1,1)
            ref_grey_videos = ref_grey_videos / 255.0
            ref_grey_videos = self.pixel_transforms(ref_grey_videos).to(dtype=torch.float32)

            frames = video_reader.get_batch(video_idx).asnumpy()
            # frame1 = video_reader.get_batch(video_idx[0]).asnumpy()
            # frame2 = video_reader.get_batch(video_idx[-1]).asnumpy()
            
            grey_frames = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]))
            for i in range(frames.shape[0]):
                # 将 RGB 转换为 BGR，因为 OpenCV 默认使用 BGR 格式
                bgr_frame = cv2.cvtColor(frames[i, ...], cv2.COLOR_RGB2BGR)
                # 将 BGR 转换为灰度
                grey_frames[i, ...] = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            grey_videos = torch.from_numpy(grey_frames).unsqueeze(1).contiguous()
            grey_videos = grey_videos.repeat(1,3,1,1)
            grey_videos = grey_videos / 255.0
            grey_videos = self.pixel_transforms(grey_videos).to(dtype=torch.float32)

            if self.use_mask:
                frame1 = frames[0]
                frame2 = frames[-1]
                camera_mask, object_mask = flow_mask(frame1, frame2,mask_video_ratio=self.mask_video_ratio)
                camera_mask = torch.tensor(camera_mask).to(dtype=torch.float32)

                camera_mask = camera_mask.unsqueeze(0).unsqueeze(0)
                camera_mask = camera_mask.repeat(self.sample_n_frames*2,4,1,1)

                # ref_mask_grey_videos = ref_grey_videos*grid_mask_camera
                # grey_mask_videos = grey_videos*grid_mask_camera

                # ref_grey_videos = torch.concat([ref_grey_videos,ref_mask_grey_videos], dim=0).to(dtype=torch.float32)
                # grey_videos = torch.concat([grey_videos,grey_mask_videos], dim=0).to(dtype=torch.float32)

        ref_videos = torch.from_numpy(video_reader.get_batch(ref_idx).asnumpy()).permute(0, 3, 1, 2).contiguous() #(N,H,W,C)->(N,C,H,W)
        videos = torch.from_numpy(video_reader.get_batch(video_idx).asnumpy()).permute(0, 3, 1, 2).contiguous() #(N,H,W,C)->(N,C,H,W)
        ref_videos = ref_videos / 255.0
        videos = videos / 255.0

        ref_videos = self.pixel_transforms(ref_videos).to(dtype=torch.float32)
        videos = self.pixel_transforms(videos).to(dtype=torch.float32)

        if self.use_grey:
            if self.use_mask:
                return file_name,videos,ref_videos,ref_grey_videos,grey_videos,camera_mask
            else: 
                return file_name,videos,ref_videos,ref_grey_videos,grey_videos
        else:
            return file_name,videos,ref_videos

    def get_file_name(self, file_path):
        return file_path.split('/')[-1].split('.')[0]
    
    @staticmethod
    def collate_fn(batch):
        # name
        name = [item['name'] for item in batch]

        # videos
        videos = [item['videos'] for item in batch]
        videos = torch.stack(videos)

        # ref_img
        ref_img = [item['ref_img'] for item in batch]
        ref_img = torch.stack(ref_img)

        # # videos
        grey_videos = [item['grey_videos'] for item in batch]
        grey_videos = torch.stack(grey_videos)

        # # ref_img
        ref_grey_img = [item['ref_grey_img'] for item in batch]
        ref_grey_img = torch.stack(ref_grey_img)

        return dict(name=name, videos=videos, ref_img=ref_img, grey_videos=grey_videos, ref_grey_img=ref_grey_img)

        # return dict(name=name, videos=videos, ref_img=ref_img)

class A2MVideoUCF(Dataset):
    def __init__(
            self, 
            video_dir: str      = '', # video dir or pkl file
            sample_size: int    = 32, 
            sample_stride: int  = 2, 
            sample_n_frames:int = 16,
            ref_drop_ratio = 0.0,
            data_frac = 1.0,
            target_fps = 8,
            use_grey = True,
        ):
        
        # Init setting
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.ref_drop_ratio = ref_drop_ratio
        self.data_frac = data_frac # 是否以一定概率drop一部分数据
        self.use_grey = use_grey
        self.target_fps = target_fps

        # Transform
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size) # (256,256)
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        metadata = pd.read_csv(
                video_dir,
                on_bad_lines="skip",
                encoding="ISO-8859-1",
                engine="python",
                sep=",",)

        if self.data_frac < 1:
            metadata = metadata.sample(frac=self.data_frac)
        metadata.dropna(inplace=True) # 使用 dropna(inplace=True) 方法删除所有包含缺失值（NaN）的行

        # Data dict
        self.metadata_list = []
        for path,label in tqdm(zip(metadata['path'],metadata['class'])):
            d = {}
            d['class'] = label
            d['path'] = path
            self.metadata_list.append(d)
        
        self.length = len(self.metadata_list)
        print(f'Total {self.length} files is available')

    def __getitem__(self, idx):
        while True:
            try:
                label = self.metadata_list[idx]['class']
                videos,ref_img,ref_grey_img,grey_videos = self.get_batch(idx)
                # file_name,videos,ref_img = self.get_batch(idx)
                break

            except Exception as e:
                # file_name = self.metadata_list[idx]['name']
                # print(file_name)
                print('error',e)
                idx = random.randint(0, self.length-1)
        sample = {
                    "ref_grey_img":ref_grey_img,
                    "grey_videos":grey_videos,
                    "videos":videos,
                    "ref_img":ref_img,
                    "label":label
                }
        return sample
    
    def __len__(self):
        return self.length

    def get_batch(self, idx):
        
        # init
        meta_data = self.metadata_list[idx]
        video_path = meta_data['path']

        # video process
        video_reader = VideoReader(video_path, ctx=cpu(0))
        video_length = len(video_reader)
        video_fps = video_reader.get_avg_fps()
        # print(video_fps,"_____________video_fps________")
        
        # sample_frames = self.sample_n_frames + 1 # refimg + videos
        # clip_length = min(video_length, (sample_frames - 1) * self.sample_stride + 1)
        # start_idx   = random.randint(0, video_length)
        # batch_index = np.linspace(start_idx, start_idx + clip_length - 1, sample_frames, dtype=int)

        batch_index = self.sample_frames_with_fps(video_length,video_fps,self.sample_n_frames+1,self.target_fps)

        if len(batch_index) != self.sample_n_frames+1:
            raise ValueError("The number of sampled frames does not match the expected count.")

        if self.use_grey:

            frames = video_reader.get_batch(batch_index).asnumpy()
            grey_frames = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]))
            for i in range(frames.shape[0]):
                # 将 RGB 转换为 BGR，因为 OpenCV 默认使用 BGR 格式
                bgr_frame = cv2.cvtColor(frames[i, ...], cv2.COLOR_RGB2BGR)
                # 将 BGR 转换为灰度
                grey_frames[i, ...] = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            grey_videos = torch.from_numpy(grey_frames).unsqueeze(1).contiguous()
            grey_videos = grey_videos.repeat(1,3,1,1)
            grey_videos = grey_videos / 255.0
            grey_videos_cache = self.pixel_transforms(grey_videos).to(dtype=torch.float32)
            grey_videos = grey_videos_cache[1:,:,:,:] # F,C,H,W
            grey_ref_img = grey_videos_cache[0,:,:,:] # C,H,W
            # grey_ref_img = grey_videos_cache[1,:,:,:] # C,H,W
            grey_ref_img = grey_ref_img.unsqueeze(0).repeat(grey_videos.shape[0],1,1,1)

        videos = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous() #(N,H,W,C)->(N,C,H,W)
        videos = videos / 255.0 

        # # ref frame
        # idx_all = np.arange(0,video_length)
        # occ_idx = np.arange(start_idx, start_idx + clip_length)
        # ref_idx = [x for x in idx_all if x not in occ_idx]
        # if len(ref_idx) == 0:
        #     ref_frame_idx = batch_index[0]
        # else:
        #     np.random.shuffle(ref_idx)
        #     ref_frame_idx = ref_idx[0]
        # ref_frame = torch.from_numpy(video_reader[ref_frame_idx].asnumpy()).permute(2, 0, 1).contiguous() 
        # ref_frame = ref_frame / 255.0
        
        # transform
        videos_cache = self.pixel_transforms(videos) # F+1,C,H,W
        videos = videos_cache[1:,:,:,:] # F,C,H,W
        ref_img = videos_cache[0,:,:,:] # C,H,W
        # ref_img = videos_cache[1,:,:,:]

        # repeat
        ref_img = ref_img.unsqueeze(0).repeat(videos.shape[0],1,1,1) # F,C,H,W


        return videos,ref_img,grey_ref_img,grey_videos

    def sample_frames_with_fps(
        self,
        total_frames,
        video_fps,
        sample_num_frames,
        sample_fps,
        start_index=None
    ):
        """sample frames proportional to the length of the frames in one second
        e.g., 1s video has 30 frames, when 'fps'=3, we sample frames with spacing of 30/3=10
        return the frame indices

        Parameters
        ----------
        total_frames : length of the video
        video_fps : original fps of the video
        sample_num_frames : number of frames to sample
        sample_fps : the fps to sample frames
        start_index : the starting frame index. If it is not None, it will be used as the starting frame index  

        Returns
        -------
        frame indices
        """
        # sample_num_frames = min(sample_num_frames, total_frames)
        interval = round(video_fps / sample_fps)
        frames_range = (sample_num_frames - 1) * interval + 1

        if start_index is not None:
            start = start_index
        elif total_frames - frames_range - 1 < 0:
            start = 0
        else:
            start = random.randint(0, total_frames - frames_range - 1)

        frame_idxs = np.linspace(
            start=start, stop=min(total_frames - 1, start + frames_range), num=sample_num_frames
        ).astype(int)

        return frame_idxs

class A2MVideoAudio(Dataset):
    def __init__(
            self, 
            video_dir:str,
            sample_size: int    = 256, 
            sample_stride: int  = 1, 
            sample_n_frames:int = 16,
        ):
        super().__init__()

        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames

        # Transform
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size) # (256,256)
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(min(sample_size)),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.classes = list(
            natsorted(p for p in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, p)))
            )
        self.classes.remove('ucfTrainTestlist')
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        # with open(video_dir, 'rb') as f:
        #     self.metadata_list = pickle.load(f)

        self.true_length = len(self.metadata_list)
        self.length = max(self.true_length,10000)
        print(f'Total {self.true_length} files is available')
    
    def __getitem__(self, idx):
        while True:
            try:
                if idx > self.true_length:
                    idx = np.random.randint(0, self.true_length)
                sample = self.get_batch(idx)
                break
            except Exception as e:
                print('error',e)
                idx = np.random.randint(0, self.true_length)

        return sample 

    def __len__(self):
        return self.length

    def get_batch(self, idx):

        """
        videos : 31,3,256,256
        ref_img : 3,256,256
        audio_feature : 30,50,384
        ref_pose : 3,256,256
        meta 
        """
        
        # init
        meta_data = self.metadata_list[idx]
        video_path = meta_data['video_path']
        whisper_path = meta_data['whisper_emb_path']

        # audio
        audio_feature = torch.load(whisper_path) 
        
        # load & check
        video_reader = VideoReader(video_path)
        video_length = min(len(video_reader),audio_feature.shape[0])
        
        # sample_frames
        sample_frames = self.sample_n_frames + 1 # self.sample_n_frames = 4, sample_frames=5
        clip_length = (sample_frames - 1) * self.sample_stride + 1 # clip_length = 9

        if clip_length > video_length :
            batch_index = np.linspace(0, clip_length - 1, sample_frames, dtype=int)
            batch_index = np.array([d for d in batch_index if d <= video_length-1],dtype=int)

            # frames
            videos = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous() #(N,H,W,C)->(N,C,H,W)
            videos = videos / 255.0
            videos = self.pixel_transforms(videos)
            ref_video = videos[0,:] # C,H,W
            gt_video = videos[1:,:] # F,C,H,W

            audios = audio_feature[batch_index]
            ref_audio = audios[0,:]  # M,D
            gt_audio = audios[1:,:]  # F,M,D

            # available length
            cur_available_length =   gt_video.shape[0]

            # pad
            pad_length = self.sample_n_frames - gt_video.shape[0]
            video_pad = torch.zeros((pad_length, *gt_video.shape[1:]), dtype=gt_video.dtype)
            gt_video = torch.cat([gt_video, video_pad], dim=0) # F,C,H,W

            audio_pad = torch.zeros((pad_length, *gt_audio.shape[1:]), dtype=gt_audio.dtype)
            gt_audio = torch.cat([gt_audio, audio_pad], dim=0)
        else:
            start_idx = np.random.randint(0, video_length - clip_length + 1)
            end_idx = start_idx + clip_length
            batch_index = np.linspace(start_idx, end_idx - 1, sample_frames, dtype=int)

            # frames
            videos = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous() #(N,H,W,C)->(N,C,H,W)
            videos = videos / 255.0
            videos = self.pixel_transforms(videos)
            ref_video = videos[0,:] # C,H,W
            gt_video = videos[1:,:] # F,C,H,W

            audios = audio_feature[batch_index]
            ref_audio = audios[0,:]  # M,D
            gt_audio = audios[1:,:]  # F,M,D

            # available length
            cur_available_length =   gt_video.shape[0]

        assert gt_video.shape[0] == self.sample_n_frames ,''+str(gt_video.shape[0])+' '+str(self.sample_n_frames)
        assert gt_audio.shape[0] == self.sample_n_frames ,''+str(gt_audio.shape[0])+' '+str(self.sample_n_frames)
        
        # mask
        mask = torch.zeros(self.sample_n_frames)
        mask[:cur_available_length] = 1
    
        # meta
        meta_ = dict(
            video_length = video_length,
            video_path = video_path,
            audio_path = whisper_path,
        )

        return dict(
            ref_video=ref_video,
            gt_video=gt_video,
            ref_audio=ref_audio,
            gt_audio=gt_audio,
            mask = mask,
            meta=meta_
        )
    
    def get_file_name(self, file_path):
        return file_path.split('/')[-1].split('.')[0]
    
    @staticmethod
    def collate_fn(batch):
        return dict(
            meta = [item["meta"] for item in batch],
            ref_video = torch.stack([item['ref_video'] for item in batch]),
            gt_video = torch.stack([item['gt_video'] for item in batch]),
            ref_audio = torch.stack([item['ref_audio'] for item in batch]),
            gt_audio = torch.stack([item['gt_audio'] for item in batch]),
            mask = torch.stack([item['mask'] for item in batch])
        )

class A2MVideoAudioPose(Dataset):
    def __init__(
            self, 
            video_dir:str,
            sample_size: int    = 256, 
            sample_stride: int  = 1, 
            sample_n_frames:int = 16,
        ):
        super().__init__()

        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames

        # Transform
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size) # (256,256)
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(min(sample_size)),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        with open(video_dir, 'rb') as f:
            self.metadata_list = pickle.load(f)
        self.true_length = len(self.metadata_list)
        self.length = max(self.true_length,10000)
        print(f'Total {self.true_length} files is available')
    
    def __getitem__(self, idx):
        while True:
            try:
                if idx > self.true_length:
                    idx = np.random.randint(0, self.true_length)
                sample = self.get_batch(idx)
                break
            except Exception as e:
                print('error',e)
                idx = np.random.randint(0, self.true_length)

        return sample 

    def __len__(self):
        return self.length

    def get_batch(self, idx):

        """
        videos : 31,3,256,256
        ref_img : 3,256,256
        audio_feature : 30,50,384
        ref_pose : 3,256,256
        meta 
        """
        
        # init
        meta_data = self.metadata_list[idx]
        video_path = meta_data['video_path']
        whisper_path = meta_data['whisper_emb_path']
        pose_path = meta_data['pose_path']

        # audio
        audio_feature = torch.load(whisper_path) 
        
        # load & check
        video_reader = VideoReader(video_path)
        pose_reader = VideoReader(pose_path)
        video_length = min(len(video_reader),audio_feature.shape[0],len(pose_reader))
        
        # sample_frames
        sample_frames = self.sample_n_frames + 1 # self.sample_n_frames = 4, sample_frames=5
        clip_length = (sample_frames - 1) * self.sample_stride + 1 # clip_length = 9

        if clip_length > video_length :
            batch_index = np.linspace(0, clip_length - 1, sample_frames, dtype=int)
            batch_index = np.array([d for d in batch_index if d <= video_length-1],dtype=int)

            # frames
            videos = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous() #(N,H,W,C)->(N,C,H,W)
            videos = videos / 255.0
            videos = self.pixel_transforms(videos)
            ref_video = videos[0,:] # C,H,W
            gt_video = videos[1:,:] # F,C,H,W

            poses = torch.from_numpy(pose_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous() #(N,H,W,C)->(N,C,H,W)
            poses = poses / 255.0
            poses = self.pixel_transforms(poses)
            ref_pose = poses[0,:] # C,H,W
            gt_pose = poses[1:,:] # F,C,H,W

            audios = audio_feature[batch_index]
            ref_audio = audios[0,:]  # M,D
            gt_audio = audios[1:,:]  # F,M,D

            # available length
            cur_available_length =   gt_video.shape[0]

            # pad
            pad_length = self.sample_n_frames - gt_video.shape[0]

            video_pad = torch.zeros((pad_length, *gt_video.shape[1:]), dtype=gt_video.dtype)
            gt_video = torch.cat([gt_video, video_pad], dim=0) # F,C,H,W

            pose_pad = torch.zeros((pad_length, *gt_pose.shape[1:]), dtype=gt_pose.dtype)
            gt_pose = torch.cat([gt_pose, pose_pad], dim=0) # F,C,H,W

            audio_pad = torch.zeros((pad_length, *gt_audio.shape[1:]), dtype=gt_audio.dtype)
            gt_audio = torch.cat([gt_audio, audio_pad], dim=0)
        else:
            start_idx = np.random.randint(0, video_length - clip_length + 1)
            end_idx = start_idx + clip_length
            batch_index = np.linspace(start_idx, end_idx - 1, sample_frames, dtype=int)

            # frames
            videos = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous() #(N,H,W,C)->(N,C,H,W)
            videos = videos / 255.0
            videos = self.pixel_transforms(videos)
            ref_video = videos[0,:] # C,H,W
            gt_video = videos[1:,:] # F,C,H,W

            poses = torch.from_numpy(pose_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous() #(N,H,W,C)->(N,C,H,W)
            poses = poses / 255.0
            poses = self.pixel_transforms(poses)
            ref_pose = poses[0,:] # C,H,W
            gt_pose = poses[1:,:] # F,C,H,W            

            audios = audio_feature[batch_index]
            ref_audio = audios[0,:]  # M,D
            gt_audio = audios[1:,:]  # F,M,D

            # available length
            cur_available_length =   gt_video.shape[0]

        assert gt_video.shape[0] == self.sample_n_frames ,''+str(gt_video.shape[0])+' '+str(self.sample_n_frames)
        assert gt_audio.shape[0] == self.sample_n_frames ,''+str(gt_audio.shape[0])+' '+str(self.sample_n_frames)
        assert gt_pose.shape[0] == self.sample_n_frames ,''+str(gt_pose.shape[0])+' '+str(self.sample_n_frames)
        
        # mask
        mask = torch.zeros(self.sample_n_frames)
        mask[:cur_available_length] = 1
    
        # meta
        meta_ = dict(
            video_length = video_length,
            video_path = video_path,
            audio_path = whisper_path,
        )

        return dict(
            ref_video=ref_video,
            gt_video=gt_video,
            ref_pose = ref_pose,
            gt_pose = gt_pose,
            ref_audio=ref_audio,
            gt_audio=gt_audio,
            mask = mask,
            meta=meta_
        )
    
    def get_file_name(self, file_path):
        return file_path.split('/')[-1].split('.')[0]
    
    @staticmethod
    def collate_fn(batch):
        return dict(
            meta = [item["meta"] for item in batch],
            ref_video = torch.stack([item['ref_video'] for item in batch]),
            gt_video = torch.stack([item['gt_video'] for item in batch]),
            ref_pose = torch.stack([item['ref_pose'] for item in batch]),
            gt_pose = torch.stack([item['gt_pose'] for item in batch]),
            ref_audio = torch.stack([item['ref_audio'] for item in batch]),
            gt_audio = torch.stack([item['gt_audio'] for item in batch]),
            mask = torch.stack([item['mask'] for item in batch])
        )

class A2MVideoAudioPoseRandomRef(Dataset):
    def __init__(
            self, 
            video_dir:str,
            sample_size: int    = 256, 
            sample_stride: int  = 1, 
            sample_n_frames:int = 16,
        ):
        super().__init__()

        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames

        # Transform
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size) # (256,256)
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(min(sample_size)),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        with open(video_dir, 'rb') as f:
            self.metadata_list = pickle.load(f)
        self.true_length = len(self.metadata_list)
        self.length = max(self.true_length,10000)
        print(f'Total {self.true_length} files is available')
    
    def __getitem__(self, idx):
        while True:
            try:
                if idx > self.true_length:
                    idx = np.random.randint(0, self.true_length)
                sample = self.get_batch(idx)
                break
            except Exception as e:
                print('error',e)
                idx = np.random.randint(0, self.true_length)

        return sample 

    def __len__(self):
        return self.length

    def get_batch(self, idx):

        """
        videos : 31,3,256,256
        ref_img : 3,256,256
        audio_feature : 30,50,384
        ref_pose : 3,256,256
        meta 
        """
        
        # init
        meta_data = self.metadata_list[idx]
        video_path = meta_data['video_path']
        whisper_path = meta_data['whisper_emb_path']
        pose_path = meta_data['pose_path']

        # audio
        audio_feature = torch.load(whisper_path) 
        
        # load & check
        video_reader = VideoReader(video_path)
        pose_reader = VideoReader(pose_path)
        video_length = min(len(video_reader),audio_feature.shape[0],len(pose_reader))
        
        # sample_frames
        sample_frames = self.sample_n_frames 
        clip_length = (sample_frames - 1) * self.sample_stride + 1 # clip_length = 9

        if clip_length > video_length :
            batch_index = np.linspace(0, clip_length - 1, sample_frames, dtype=int)
            batch_index = list(np.array([d for d in batch_index if d <= video_length-1],dtype=int))

            # ref idx
            idx_all = np.arange(0,video_length)
            start_idx = 0
            occ_idx = np.arange(start_idx, start_idx + clip_length)
            ref_idx = [x for x in idx_all if x not in occ_idx]
            if len(ref_idx) == 0:
                ref_frame_idx = batch_index[0]
            else:
                np.random.shuffle(ref_idx)
                ref_frame_idx = ref_idx[0]
            batch_index = [ref_frame_idx] + batch_index

            # frames
            videos = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous() #(N,H,W,C)->(N,C,H,W)
            videos = videos / 255.0
            videos = self.pixel_transforms(videos)
            ref_video = videos[0,:] # C,H,W
            gt_video = videos[1:,:] # F,C,H,W

            poses = torch.from_numpy(pose_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous() #(N,H,W,C)->(N,C,H,W)
            poses = poses / 255.0
            poses = self.pixel_transforms(poses)
            ref_pose = poses[0,:] # C,H,W
            gt_pose = poses[1:,:] # F,C,H,W

            audios = audio_feature[batch_index]
            ref_audio = audios[0,:]  # M,D
            gt_audio = audios[1:,:]  # F,M,D

            # available length
            cur_available_length =   gt_video.shape[0]

            # pad
            pad_length = self.sample_n_frames - gt_video.shape[0]

            video_pad = torch.zeros((pad_length, *gt_video.shape[1:]), dtype=gt_video.dtype)
            gt_video = torch.cat([gt_video, video_pad], dim=0) # F,C,H,W

            pose_pad = torch.zeros((pad_length, *gt_pose.shape[1:]), dtype=gt_pose.dtype)
            gt_pose = torch.cat([gt_pose, pose_pad], dim=0) # F,C,H,W

            audio_pad = torch.zeros((pad_length, *gt_audio.shape[1:]), dtype=gt_audio.dtype)
            gt_audio = torch.cat([gt_audio, audio_pad], dim=0)
        else:
            start_idx = np.random.randint(0, video_length - clip_length + 1)
            end_idx = start_idx + clip_length
            batch_index = list(np.linspace(start_idx, end_idx - 1, sample_frames, dtype=int))

            # ref index
            idx_all = np.arange(0,video_length)
            occ_idx = np.arange(start_idx, start_idx + clip_length)
            ref_idx = [x for x in idx_all if x not in occ_idx]
            if len(ref_idx) == 0:
                ref_frame_idx = batch_index[0]
            else:
                np.random.shuffle(ref_idx)
                ref_frame_idx = ref_idx[0]

            batch_index = [ref_frame_idx] + batch_index

            # frames
            videos = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous() #(N,H,W,C)->(N,C,H,W)
            videos = videos / 255.0
            videos = self.pixel_transforms(videos)
            ref_video = videos[0,:] # C,H,W
            gt_video = videos[1:,:] # F,C,H,W

            poses = torch.from_numpy(pose_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous() #(N,H,W,C)->(N,C,H,W)
            poses = poses / 255.0
            poses = self.pixel_transforms(poses)
            ref_pose = poses[0,:] # C,H,W
            gt_pose = poses[1:,:] # F,C,H,W            

            audios = audio_feature[batch_index]
            ref_audio = audios[0,:]  # M,D
            gt_audio = audios[1:,:]  # F,M,D

            # available length
            cur_available_length =   gt_video.shape[0]

        assert gt_video.shape[0] == self.sample_n_frames ,''+str(gt_video.shape[0])+' '+str(self.sample_n_frames)
        assert gt_audio.shape[0] == self.sample_n_frames ,''+str(gt_audio.shape[0])+' '+str(self.sample_n_frames)
        assert gt_pose.shape[0] == self.sample_n_frames ,''+str(gt_pose.shape[0])+' '+str(self.sample_n_frames)
        
        # mask
        mask = torch.zeros(self.sample_n_frames)
        mask[:cur_available_length] = 1
    
        # meta
        meta_ = dict(
            video_length = video_length,
            video_path = video_path,
            audio_path = whisper_path,
        )

        return dict(
            ref_video=ref_video,
            gt_video=gt_video,
            ref_pose = ref_pose,
            gt_pose = gt_pose,
            ref_audio=ref_audio,
            gt_audio=gt_audio,
            mask = mask,
            meta=meta_
        )
    
    def get_file_name(self, file_path):
        return file_path.split('/')[-1].split('.')[0]
    
    @staticmethod
    def collate_fn(batch):
        return dict(
            meta = [item["meta"] for item in batch],
            ref_video = torch.stack([item['ref_video'] for item in batch]),
            gt_video = torch.stack([item['gt_video'] for item in batch]),
            ref_pose = torch.stack([item['ref_pose'] for item in batch]),
            gt_pose = torch.stack([item['gt_pose'] for item in batch]),
            ref_audio = torch.stack([item['ref_audio'] for item in batch]),
            gt_audio = torch.stack([item['gt_audio'] for item in batch]),
            mask = torch.stack([item['mask'] for item in batch])
        )


def generate_non_equal_random_lists(frame_num,sample_num):
    list1 = [np.random.randint(0, frame_num) for _ in range(sample_num)]

    list2 = []
    for i in range(len(list1)):
        available_numbers = list(range(0, list1[i])) + list(range(list1[i] + 1, frame_num))
        list2.append(random.choice(available_numbers))

    return list1, list2

def flow_mask(frame1, frame2, l_window_size=128, s_window_size=32, direction_var_threshold=6, direction_threshold=0.4, mask_video_ratio=0.5, name=None):

    # frame1 = video_reader.get_batch(batch_index).asnumpy()[0]
    # frame2 = video_reader.get_batch(end_index).asnumpy()[0]

    frame1 = cv2.resize(frame1, (256, 256), interpolation=cv2.INTER_LINEAR)
    frame2 = cv2.resize(frame2, (256, 256), interpolation=cv2.INTER_LINEAR)

    # for i in range(frames.shape[0]):
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)

    # 灰度化与高斯滤波
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (5,5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5,5), 0)


    # 计算Farneback光流
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 
        pyr_scale=0.5, levels=3, winsize=30, 
        iterations=3, poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    # 提取光流矢量场
    u = flow[...,0]
    v = flow[...,1]
    # 计算光流方向与幅度
    magnitude = np.sqrt(u**2 + v**2)
    direction = np.arctan2(v, u)  # 方向弧度值 [-π, π]
    
    # 新增参数
    # LARGE_WINDOW = 128  # 大窗口尺寸
    height, width = u.shape
    DIRECTION_THRESHOLD = np.pi/6  # 方向差异阈值 (30度)

    # 计算大窗口的基准运动方向
    large_window_directions = np.zeros((height//l_window_size + 1, width//l_window_size + 1))
    
    # Step 1: 计算每个大窗口的平均方向
    for y_large in range(0, height, l_window_size):
        for x_large in range(0, width, l_window_size):
            # 提取大窗口区域
            win_u_large = u[y_large:y_large+l_window_size, x_large:x_large+l_window_size]
            win_v_large = v[y_large:y_large+l_window_size, x_large:x_large+l_window_size]
            
            # 计算平均方向（矢量平均）
            avg_u = np.mean(win_u_large)
            avg_v = np.mean(win_v_large)
            avg_direction = np.arctan2(avg_v, avg_u)
            
            large_window_directions[y_large//l_window_size, x_large//l_window_size] = avg_direction

    # Step 3: 小窗口处理 255是选，0是不选
    grid_mask_camera = np.ones((height, width), dtype=np.uint8) * 255  # 初始化为全白
    grid_mask_object = np.ones((height, width), dtype=np.uint8) * 255  # 初始化为全白


    for y in range(0, height, s_window_size):
        for x in range(0, width, s_window_size):
            
            # 获取所属大窗口索引
            large_row = y // l_window_size
            large_col = x // l_window_size
            base_direction = large_window_directions[large_row, large_col]

            # 提取当前窗口的光流数据
            win_u = u[y:y+s_window_size, x:x+s_window_size]
            win_v = v[y:y+s_window_size, x:x+s_window_size]
            win_mag = magnitude[y:y+s_window_size, x:x+s_window_size]
            win_dir = direction[y:y+s_window_size, x:x+s_window_size]

            # # 判断条件1：窗口内无显著运动
            # avg_mag = np.mean(win_mag)
            # if avg_mag < flow_mag_threshold:
            #     grid_mask_camera[y:y+window_size, x:x+window_size] = 0  # 掩膜为黑色
            #     grid_mask_object[y:y+window_size, x:x+window_size] = 0  # 掩膜为黑色
            #     continue

            # 判断条件2：大小窗口方向一致性
            direction_diff = np.abs(win_dir - base_direction)
            direction_diff = np.minimum(direction_diff, 2*np.pi - direction_diff)  # 处理圆周差
            inconsistent_ratio = np.mean(direction_diff > DIRECTION_THRESHOLD)
            
            if inconsistent_ratio > direction_threshold:
                # grid_mask_camera[y:y+s_window_size, x:x+s_window_size] = 255  # 相机区域重新置白，取
                grid_mask_camera[y:y+s_window_size, x:x+s_window_size] = 0  # 去除主体运动区域
            else:
                grid_mask_object[y:y+s_window_size, x:x+s_window_size] = 0  # 去除相机运动区域

            # 判断条件3：方向差异性（方差）
            dir_variance = np.var(win_dir)
            if dir_variance > direction_var_threshold:
                grid_mask_camera[y:y+s_window_size, x:x+s_window_size] = 0  # 去除快速运动的大主体
            else :
                grid_mask_object[y:y+s_window_size, x:x+s_window_size] = 0  # 去除不动的背景

            if dir_variance < 0.2:
                grid_mask_camera[y:y+s_window_size, x:x+s_window_size] = 255  # 将不动的背景保留

    # Step 4: 形态学处理优化掩膜
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    grid_mask_camera = cv2.morphologyEx(grid_mask_camera, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite(f"/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_diff/windows_{s_window_size}/{name}_grid_mask_camera.jpg",grid_mask_camera)
    grid_mask_object = cv2.morphologyEx(grid_mask_object, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite(f"/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_diff/windows_{s_window_size}/{name}_grid_mask_object.jpg",grid_mask_object)

    h,w = grid_mask_camera.shape
    # 1. 统计所有全0窗口的位置
    white_windows = []
    for y in range(0, h, s_window_size):
        for x in range(0, w, s_window_size):
            window = grid_mask_camera[y:y+s_window_size, x:x+s_window_size]
            if np.all(window == 255):  # 检查窗口是否全0
                white_windows.append((y, x))

    # 2. 计算需要保留的窗口数（32个）
    max_white_windows = int(((h / s_window_size)**2)*(1-mask_video_ratio))  # 32*32*32=32768（50%）
    current_white_count = len(white_windows)

    if current_white_count > max_white_windows:
        # 3. 随机选择保留的窗口
        np.random.shuffle(white_windows)
        keep_windows = white_windows[:max_white_windows]
        
        # 4. 将未选中的全0窗口改为全255
        for y, x in white_windows[max_white_windows:]:
            grid_mask_camera[y:y+s_window_size, x:x+s_window_size] = 0

    grid_mask_camera = grid_mask_camera / 255
    grid_mask_object = grid_mask_object / 255

    step = grid_mask_camera.shape[0] // 32
    grid_mask_camera = grid_mask_camera[::step,::step]
    grid_mask_object = grid_mask_object[::step,::step]

    return grid_mask_camera, grid_mask_object


if __name__ == "__main__":
    # dataset = CelebvText()
    
    
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16,)
    # for idx, batch in enumerate(dataloader):
        # print(batch["videos"].shape, len(batch["text"]))
        # for i in range(batch["videos"].shape[0]):
            # save_videos_grid(batch["videos"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)

    from torch.utils.data import DataLoader
    # dataset = AMDVideoAudioFeature(
    #     path=data_path,
    #     path_type="file",
    #     motion_seq_len=motion_seq_len,
    #     sample_n_frames=num_frames,
    #     audio_processor=audio_processor
    # )
    dataset = AMDVideoAudioFeature(
        video_dir = '/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/code/AMD_linear/dataset/path/lhz/train.pkl',
        path_type = 'file'
    )
    dataloader = DataLoader(
        dataset,2,True,num_workers=0,
        collate_fn=dataset.collate_fn
    )

    # dataloader = DataLoader(dataset,batch_size=2,collate_fn=dataset.collate_fn,num_workers=2)
    
    # d = dataset[10]
    # video = d["videos"]
    # audio = d["audio_feature"]
    # refimg = d["ref_img"]
    for d in dataloader:
        video = d["videos"]
        audio_feature = d["audio_feature"]
        refimg = d["ref_img"]
        # break
    print(video.shape)
    print(audio_feature.shape)




# 