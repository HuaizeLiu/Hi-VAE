import torch
from einops import rearrange
from moviepy.editor import VideoFileClip
import numpy as np
from moviepy.editor import (VideoFileClip,
                            AudioFileClip,
                            concatenate_videoclips,
                            CompositeAudioClip,
                            ImageSequenceClip,
                            )
from moviepy.audio.AudioClip import AudioArrayClip
def read_frames(video_path:str):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    frames = np.array(list(video_clip.iter_frames()))
    video_fps = video_clip.fps
    return {
        "video": frames,
        "fps": video_fps,
        "audio_clip": audio_clip
    }
def audio2array(audio_clip,start_frame,end_frame,video_fps,sampling_rate,audio_len):
    audio_fps = audio_clip.fps 
    start_time = start_frame / video_fps
    end_time = end_frame / video_fps
    sampled_audio = audio_clip.subclip(start_time, end_time)
    sampled_audio = sampled_audio.set_fps(sampling_rate) 
    audio_chunks = list(sampled_audio.iter_chunks(fps=sampling_rate,chunksize=50000))
    audio = np.concatenate(audio_chunks, axis=0)
    if len(audio.shape) == 2:
        audio = np.mean(audio,axis=1)
    audio = np.interp(np.arange(0,audio_len),np.arange(0,len(audio)),audio)
    return {
        "audio": audio,
        "fps": audio_fps
    }
def read_video(video_path,sampling_rate:int = 16000,frame_idx=None):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio

    frames = np.array(list(video_clip.iter_frames()))
    video_fps = video_clip.fps
    audio_fps = audio_clip.fps 
    if frame_idx is not None:
        start_time = frame_idx[0] / video_fps
        end_time = frame_idx[-1] / video_fps
    else:
        start_time = 0
        end_time = len(frames)  / video_fps
    sampled_audio = audio_clip.subclip(start_time, end_time)
    sampled_audio = sampled_audio.set_fps(sampling_rate) 
    audio_chunks = list(sampled_audio.iter_chunks(fps=sampling_rate,chunksize=50000))
    audio = np.concatenate(audio_chunks, axis=0)
    if len(audio.shape) == 2:
        audio = np.mean(audio,axis=1)
    return { 
        "video": frames,
        "audio": audio,
        "video_fps" : video_fps,
        "audio_fps": audio_fps
    } 
def write_video(output_path, frames, audio, video_fps, audio_fps):
    video_clip = ImageSequenceClip(list(frames), fps=video_fps)
    audio_clip = AudioArrayClip(audio,audio_fps)
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
def tensor2frames(x:torch.Tensor):
    if len(x.shape) == 5:
        x = rearrange(x,'b t c h w -> (b t) c h w')
    frames = []
    for image in x:
        image = image.permute(1, 2, 0)
        image_np = ((image / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()
        frames.append(image_np)
    return frames
if __name__ == "__main__":
    video_path = "/mnt/spaceai-data/tts/team/digital_avatar_group/fenghe/datasets/celebv-hq/-3Bl8i34Z7Q_0.mp4"
    d = read_frames(video_path)
    print(audio2array(d["audio_clip"],0,16,24,16000,10000))
    # video = VideoFileClip(video_path)
    # print(len(list(video.iter_frames())))