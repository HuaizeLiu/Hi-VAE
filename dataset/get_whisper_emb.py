# pylint: disable=W1203,W0718
"""
This module is used to process videos to prepare data for training. It utilizes various libraries and models
to perform tasks such as video frame extraction, audio extraction, face mask generation, and face embedding extraction.
The script takes in command-line arguments to specify the input and output directories, GPU status, level of parallelism,
and rank for distributed processing.

Usage:
    python -m scripts.data_preprocess --input_dir /path/to/video_dir --dataset_name dataset_name --gpu_status --parallelism 4 --rank 0

Example:
    python -m scripts.data_preprocess -i data/videos -o data/output -g -p 4 -r 0
"""
import argparse
import logging
import os,sys
import json
from pathlib import Path
from typing import List

import cv2
import torch
from tqdm import tqdm

sys.path.append(os.path.split(sys.path[0])[0])
print(sys.path)

from hallo.datasets.audio_processor import AudioProcessor, WhisperAudioProcessor
from hallo.datasets.image_processor import ImageProcessorForDataProcessing
from hallo.utils.util import convert_video_to_images, extract_audio_from_videos



# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def setup_directories(video_path: Path) -> dict:
    """
    Setup directories for storing processed files.

    Args:
        video_path (Path): Path to the video file.

    Returns:
        dict: A dictionary containing paths for various directories.
    """
    base_dir = video_path.parent.parent
    dirs = {
        "face_mask": base_dir / "face_mask",
        "sep_pose_mask": base_dir / "sep_pose_mask",
        "sep_face_mask": base_dir / "sep_face_mask",
        "sep_lip_mask": base_dir / "sep_lip_mask",
        "face_emb": base_dir / "face_emb",
        "audio_emb": base_dir / "audio_emb",
        "mel_emb": base_dir / "mel_emb",
        "whisper_audio_emb": base_dir / "whisper_audio_emb",
        "whisper_mel_emb": base_dir / "whisper_mel_emb",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def process_single_video(video_path: Path,
                         output_dir: Path,
                         audio_processor: AudioProcessor,
                         extract_audio: bool,
                         vid_list) -> None:
    """
    Process a single video file.

    Args:
        video_path (Path): Path to the video file.
        output_dir (Path): Directory to save the output.
        image_processor (ImageProcessorForDataProcessing): Image processor object.
        audio_processor (AudioProcessor): Audio processor object.
        gpu_status (bool): Whether to use GPU for processing.
    """
    assert video_path.exists(), f"Video path {video_path} does not exist"
    dirs = setup_directories(video_path)
    logging.info(f"Processing video: {video_path}")

    
    if extract_audio:
        audio_output_dir = output_dir / 'audios'
        audio_output_dir.mkdir(parents=True, exist_ok=True)
        audio_output_path = audio_output_dir / f'{video_path.stem}.wav'
        audio_output_path = extract_audio_from_videos(
                video_path, audio_output_path)
        logging.info(f"Audio extracted to: {audio_output_path}")

        print(video_path,"process")
    try:   
        audio_path = output_dir / "audios" / f"{video_path.stem}.wav"
        if vid_list :
            vid_basename = os.path.splitext(os.path.basename(video_path))[0]
            if vid_basename in vid_list:
                print(video_path,"pase")
                return
        whisper_chunks, mel_feature = audio_processor.preprocess(audio_path)
        torch.save(whisper_chunks, str(
                dirs["whisper_audio_emb"] / f"{video_path.stem}.pt"))
        
        torch.save(mel_feature, str(
                dirs["whisper_mel_emb"] / f"{video_path.stem}.pt"))
                # audio_emb, _ = audio_processor.preprocess(audio_path)
                # torch.save(audio_emb, str(
                #     dirs["audio_emb"] / f"{video_path.stem}.pt"))
                # mel_emb = audio_processor.mel_preprocess(audio_path)
                # torch.save(mel_emb, str(
                #     dirs["mel_emb"] / f"{video_path.stem}.pt"))
        print(video_path,"process")
        
    except Exception as e:
        logging.error(f"Failed to process video {video_path}: {e}")


def process_all_videos(input_video_list: List[Path], output_dir: Path, data_meta_path: Path, extract_audio: bool) -> None:
    """
    Process all videos in the input list.

    Args:
        input_video_list (List[Path]): List of video paths to process.
        output_dir (Path): Directory to save the output.
        gpu_status (bool): Whether to use GPU for processing.
    """
    audio_separator_model_file = "/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/liuhuaize/hallo/pretrained_models/audio_separator/Kim_Vocal_2.onnx"
    whisper_model_path = '/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/liuhuaize/hallo/pretrained_models/whisper/whisper_tiny.pt'
    
    audio_processor = WhisperAudioProcessor(
        16000,
        30,
        whisper_model_path,
        os.path.dirname(audio_separator_model_file),
        os.path.basename(audio_separator_model_file),
        os.path.join(output_dir, "vocals"),
    )
    
    # 读取处理好的json文件
    vid_list = []
    if data_meta_path and os.path.isfile(data_meta_path):      
        with open(data_meta_path, "r", encoding="utf-8") as f:
            vid_json_list = json.load(f)
            for vid in vid_json_list:
                vid_list.append(os.path.splitext(os.path.basename(vid["video_path"]))[0])

    for video_path in tqdm(input_video_list, desc="Processing videos"):
        process_single_video(video_path, output_dir,
                             audio_processor, extract_audio, vid_list)


def get_video_paths(source_dir: Path, parallelism: int, rank: int) -> List[Path]:
    """
    Get paths of videos to process, partitioned for parallel processing.

    Args:
        source_dir (Path): Source directory containing videos.
        parallelism (int): Level of parallelism.
        rank (int): Rank for distributed processing.

    Returns:
        List[Path]: List of video paths to process.
    """
    video_paths = [item for item in sorted(
        source_dir.iterdir()) if item.is_file() and item.suffix == '.mp4']
    return [video_paths[i] for i in range(len(video_paths)) if i % parallelism == rank]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process videos to prepare data for training. Run this script twice with different GPU status parameters."
    )
    parser.add_argument("-i", "--input_dir", type=Path,default="/mnt/pfs-mc0p4k/tts/team/didonglin/lhz/hallo/dataset_test_2/videos",
                        required=False, help="Directory containing videos")
    parser.add_argument("-o", "--output_dir", type=Path,
                        help="Directory to save results, default is parent dir of input dir")
    parser.add_argument("-d", "--data_meta_path", type=Path,default="",
                        help="Directory to save results, default is parent dir of input dir")
    parser.add_argument("-s", "--step", type=int, default=2,
                        help="Specify data processing step 1 or 2, you should run 1 and 2 sequently")
    parser.add_argument("-p", "--parallelism", default=1,
                        type=int, help="Level of parallelism")
    parser.add_argument("-r", "--rank", default=0, type=int,
                        help="Rank for distributed processing")
    parser.add_argument("-e", "--extract_audio", default=False, type=bool,
                        help="wheather extract audio from video")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir.parent

    para_video_path_list = get_video_paths(
        args.input_dir, args.parallelism, args.rank)

    # args.data_meta_path = "/mnt/lpai-dione/ssai/cvg/team/didonglin/lhz/hallo/data/dataset_test_stage1.json"

    if not para_video_path_list:
        logging.warning("No videos to process.")
    else:
        process_all_videos(para_video_path_list, args.output_dir, args.data_meta_path, extract_audio=args.extract_audio)
