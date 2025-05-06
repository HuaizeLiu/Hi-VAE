import glob
import os
import pickle
import random
from tqdm import tqdm
import imageio.v3 as iio
import os
from multiprocessing import Pool


def split_video_whisper():
    def process_video_whisper(videodir,audiodir):
        data_list = []
        video_files = glob.glob(os.path.join(videodir, '**', '*.mp4'), recursive=True)
        
        for f in tqdm(video_files):
            name = os.path.basename(f).split('.')[0]
            cur_whisper_path = os.path.join(audiodir,f'{name}.pt')
            if os.path.exists(cur_whisper_path):
                d = {"video_path":f,"whisper_emb_path":cur_whisper_path}
                data_list.append(d)

        print(len(data_list))
        return data_list

    data = []
    data += process_video_whisper('/mnt/pfs-gv8sxa/tts/dhg/zqy/data/celeb/videos','/mnt/pfs-gv8sxa/tts/dhg/zqy/data/celeb/whisper_embs')
    data += process_video_whisper('/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/liuhuaize/dataset/FaceVid_240h/videos','/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/liuhuaize/dataset/FaceVid_240h/whisper_embs')
    print(f'Total num of data:{len(data)}')
    save_dir = '/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/code/AMD2/dataset/path/video_whisper_pose'
    eval_data = data[:100]
    train_data = data[100:]
    with open('./path/video_whisper_pose/train.pkl', 'wb') as file:
        # 使用pickle模块的dump方法保存数据
        pickle.dump(train_data, file)

    with open('./path/video_whisper_pose/eval.pkl', 'wb') as file:
        # 使用pickle模块的dump方法保存数据
        pickle.dump(eval_data, file)

def AMD_process_video(videodir = None):
    if videodir is None :
        video_dir = './train_data_amd.txt'
    else:
        video_dir = videodir

    with open(video_dir, 'r') as file:
        lines = file.readlines()
    video_dirs = [line.strip() for line in lines]

    train_files = []
    eval_files = []
    for dir in video_dirs:
        cur_files = glob.glob(os.path.join(dir, '**', '*.mp4'), recursive=True)
        random.shuffle(cur_files)
        eval_files += cur_files[:20]
        train_files += cur_files[20:]

    print(f'Total {len(train_files)} !!!')
    print(f'Total {len(eval_files)} !!!')

    with open('./path/amd2/train.pkl', 'wb') as file:
        pickle.dump(train_files, file)

    with open('./path/amd2/eval.pkl', 'wb') as file:
        pickle.dump(eval_files, file)

def A2M_process_video(videodir = None):
    if videodir is None :
        video_dir = './train_data_a2m.txt'
    else:
        video_dir = videodir

    with open(video_dir, 'r') as file:
        lines = file.readlines()
    video_dirs = [line.strip() for line in lines]

    train_files = []
    eval_files = []
    for dir in video_dirs:
        cur_datalist = []
        cur_files = glob.glob(os.path.join(dir, '**', '*.mp4'), recursive=True)
        for f in tqdm(cur_files):
            whisper_path = f.replace('videos','whisper_embs').replace('.mp4','.pt')
            pose_path = f.replace('videos','videos_dwpose')
            if os.path.exists(whisper_path) and os.path.exists(pose_path):
                cur_datalist.append({'video_path':f,'whisper_emb_path':whisper_path,'pose_path':pose_path})
        random.shuffle(cur_datalist)
        eval_files += cur_datalist[:20]
        train_files += cur_datalist[20:]

        print(dir)
        print(len(cur_datalist))

    print(f'Total {len(train_files)} !!!')
    print(f'Total {len(eval_files)} !!!')

    with open('./path/a2m/train.pkl', 'wb') as file:
        pickle.dump(train_files, file)

    with open('./path/a2m/eval.pkl', 'wb') as file:
        pickle.dump(eval_files, file)

def create_video_from_frames(frame_folder, output_video_path, fps=30):
    # 获取所有图片路径，并按名称排序以保持顺序
    frame_files = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.png')])
    
    # 读取并存储帧
    frames = [iio.imread(f) for f in frame_files]
    
    # 将帧写入视频文件
    iio.imwrite(output_video_path, frames, fps=fps)


def fullbody_imgs2video_multiprocess(datadir,output_dir):
    pool = Pool(64)
    images_folders = []
    
    data_dirs = [os.path.join(datadir,d) for d in os.listdir(datadir)]
    for data in tqdm(data_dirs):
        name = os.path.basename(data)
        frame_folder = os.path.join(data,'content','images')
        video_out_path = os.path.join(output_dir,f'{name}.mp4')

        pool.apply_async(create_video_from_frames, args=(frame_folder,video_out_path))
    
    
    pool.close()

    pool.join()

    print("结束主进程……")


def fullbody_imgs2video(datadir,output_dir):
    pool = Pool(64)
    source_dir = '/mnt/pfs-gv8sxa/tts/dhg/zqy/data/dataset_0703/datasets_tiktok_0703_clean'
    target_dir = '/mnt/pfs-gv8sxa/tts/dhg/zqy/data/full_body/dwpose'

    source_name = [n for n in os.listdir(source_dir)]
    target_name = [n.split('.')[0] for n in os.listdir(target_dir)]

    print(f'source {len(source_name)}')
    print(f'target {len(target_name)}')

    set_s = set(source_name)
    set_t = set(target_name)
    need_name = list(set_s-set_t)
    
    data_dirs = [os.path.join(datadir,n) for n in need_name]
    for data in tqdm(data_dirs):
        name = os.path.basename(data)
        frame_folder = os.path.join(data,'content','dwpose')
        video_out_path = os.path.join(output_dir,f'{name}.mp4')

        pool.apply_async(create_video_from_frames, args=(frame_folder,video_out_path))
    
    
    pool.close()

    pool.join()

    print("结束主进程……")

def fullbody_video():
    datadir = '/mnt/pfs-gv8sxa/tts/dhg/zqy/data/full_body/video'
    video_files = glob.glob(os.path.join(datadir, '**', '*.mp4'), recursive=True)
    cur_files = [f for f in video_files if 'TIKTOK' not in f]
    print(len(cur_files))
    train_files = cur_files
    eval_files = cur_files[:100]

    with open('./path/amd_fullbody/train.pkl', 'wb') as file:
        pickle.dump(train_files, file)

    with open('./path/amd_fullbody/eval.pkl', 'wb') as file:
        pickle.dump(eval_files, file)

def test_data_time():
    import time
    from tqdm import tqdm
    from decord import VideoReader
    from decord import cpu, gpu
    import glob
    import os
    datalist = []


    video_dir = '/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/code/AMD2/dataset/train_data_amd.txt'

    with open(video_dir, 'r') as file:
        lines = file.readlines()
    video_dirs = [line.strip() for line in lines]

    files = []
    for dir in video_dirs:
        cur_files = glob.glob(os.path.join(dir, '**', '*.mp4'), recursive=True)
        files += cur_files

    for f in tqdm(files):
        try:
            start_time = time.time()  # 记录开始时间

            test_idx = [1,2,3,4,5,6,7]
            video_reader = VideoReader(f, ctx=cpu(0))
            video_length = len(video_reader)
            if video_length < 10 :
                continue
            video = video_reader.get_batch(test_idx)

            end_time = time.time()   # 记录结束时间

            execution_time = end_time - start_time
            if execution_time <= 3.0:
                datalist.append(f)
            else:
                print(f'{f} # Time:{execution_time} # Frame:{video_length}')
        except Exception as e:
            continue

    with open('./amd_available_data.pkl', 'wb') as file:
        pickle.dump(datalist, file)


def get_split_video_motion_path_pkl(video_path_list:list,output_dir:str):

    def split_list_into_n(lst, n=8):
        if n > len(lst):
            n = len(lst)  # 如果 n 大于列表长度，则设置 n 为列表长度
        k, m = divmod(len(lst), n)

        result = []
        for i in range(n):
            start_index = i * k + min(i, m)
            end_index = (i + 1) * k + min(i + 1, m)
            result.append(lst[start_index:end_index])
        
        return result

    datas = [] 
    for f in tqdm(video_path_list):
        name = os.path.basename(f).split('.')[0]
        motion_dir = os.path.join(os.path.dirname(os.path.dirname(f)),'motion')
        motion_path = os.path.join(motion_dir,f'{name}.pt')

        datas.append({'video_path':f,'motion_path':motion_path})

    print(f'total avail data:{len(datas)}')

    # split
    split_data_list = split_list_into_n(datas,8)

    for i,f in enumerate(split_data_list):
        output_path = os.path.join(output_dir,f'{i}.pkl')
        print(f'send data to {output_path},len:{len(f)}')
        with open(output_path, 'wb') as file:
            pickle.dump(f, file)



if __name__ == '__main__':
    # data_dir = '/mnt/pfs-gv8sxa/tts/dhg/zqy/data/dataset_0703/datasets_tiktok_0703_clean'
    # output_dir = '/mnt/pfs-gv8sxa/tts/dhg/zqy/data/full_body/dwpose'
    # fullbody_imgs2video(data_dir,output_dir)

    # # AMD Video
    AMD_process_video()


    # video_dir = '/mnt/pfs-gv8sxa/tts/dhg/zqy/data/celeb/whisper_embs'
    # video_files = os.listdir(video_dir)
    # print(len(video_files))

    # split_video_whisper()
       

    # # ------------------ test data time ---------
    # test_data_time()

    # # ------------------ split video motion data ---------
    # video_audio_dir = ['/mnt/pfs-gv8sxa/tts/dhg/zqy/data/celeb/videos',
    #                    '/mnt/pfs-gv8sxa/tts/dhg/zqy/data/Asian_emo/videos',
    #                    '/mnt/pfs-gv8sxa/tts/dhg/zqy/data/FaceVid_240h/videos',
    #                    '/mnt/pfs-gv8sxa/tts/dhg/zqy/data/mead/videos',
    #                    '/mnt/pfs-gv8sxa/tts/dhg/zqy/data/hdtf_dit/videos']
    # video_path_list = []
    # for directory in video_audio_dir:
    #     cur_video_files = glob.glob(os.path.join(directory, '**', '*.mp4'), recursive=True)
    #     video_path_list += cur_video_files
    # print('一共有这么多视频',len(video_path_list))
    # output_dir = '/mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/dataset/path/a2m/videos_pkl'
    # get_split_video_motion_path_pkl(video_path_list,output_dir)

    # # -------------------- A2M video -------------------
    # A2M_process_video()

    