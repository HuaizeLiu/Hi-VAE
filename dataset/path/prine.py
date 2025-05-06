import glob
import os
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

min_video_len = 31
def process_video(video_path):
    try:
        video = VideoFileClip(video_path)
        length = int(video.fps * video.duration)
        if length > min_video_len:
            cnt = 0
            for _ in video.iter_frames():
                cnt = cnt + 1
                if cnt > min_video_len:
                    return video_path
        return None
    except:
        return None
pkls = glob.glob("/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/code/AMD_linear/dataset/path/**/*.pkl",recursive=True)
for pkl in pkls:
    print(f"pruning {pkl}")
    res_list = []
    with open(pkl,"rb") as f:
        videos = pickle.load(f)
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_video, video_path) for video_path in videos]
        cnt = 0
        for future in tqdm(as_completed(futures), total=len(videos)):
            video_path = future.result()
            if video_path:
                cnt += 1
                res_list.append(video_path) 
        print(f"{pkl} has {cnt} valid videos")
    save_path = os.path.join(os.path.dirname(pkl),os.path.basename(pkl).replace(".pkl","_pruned.pkl"))
    with open(save_path,"wb") as g:
        pickle.dump (res_list,g)