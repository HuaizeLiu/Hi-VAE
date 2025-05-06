import cv2

def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return -1

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def plot(data):
    import matplotlib.pyplot as plt
    
    if data is None:
        data = [24, 35, 12, 34]  # 请用你的完整列表替换这里

    indices = range(len(data))

    plt.bar(indices, data)
    plt.xticks(indices)
    plt.title('Bar Chart of List Data')
    plt.xlabel('Index')
    plt.ylabel('Value')

    plt.show()


if __name__ == '__main__':
    # import glob
    # import os
    # from tqdm import tqdm
    # video_dir= '/mnt/pfs-gv8sxa/tts/dhg/zqy/data/celeb/videos'
    # video_files = glob.glob(os.path.join(video_dir, '**', '*.mp4'), recursive=True) 
    # d = [0]*1025
    # for f in tqdm(video_files):
    #     cur_cnt = get_frame_count(f)
    #     cur_cnt = 1024 if cur_cnt>1024 else  cur_cnt
    #     print(cur_cnt)
    #     d[cur_cnt] +=1
    #     break

    plot(None)