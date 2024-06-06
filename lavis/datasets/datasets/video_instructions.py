import os
import json

from PIL import Image

from collections import OrderedDict
from pathlib import Path

from lavis.datasets.datasets.minigpt4qwen_datasets import Minigpt4QwenDataset

from moviepy.editor import VideoFileClip
import cv2

import random

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image_id"]+'.jpg',
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )

def extract_frames(video_path, num_frames):
    # get the total number of frames
    clip = VideoFileClip(video_path)
    total_num_frames = int(clip.duration * clip.fps)
    clip.close()

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    sampling_interval = int(total_num_frames / num_frames)

    if sampling_interval == 0: # total_frames < target_frames, 逐帧提取
        sampling_interval = 1
    
    images = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sampling_interval == 0:
            frame = frame[:,:,::-1]# BGR to RGB
            images.append(Image.fromarray(frame).convert("RGB"))
        frame_count += 1
        if len(images) >= num_frames:
            break
    cap.release()

    if len(images) ==0:
        raise AssertionError(f"Video not found: {video_path}")
    
    return images


# class InstructionDataset(Minigpt4QwenDataset, __DisplMixin):
class VideoInstructionDataset(Minigpt4QwenDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, max_frames):
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r")))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.max_frames = max_frames

        self._add_instance_ids()

    def __getitem__(self, index):
        try:
            ann = self.annotation[index]

            video_path = os.path.join(self.vis_root,ann['video'])
            images = extract_frames(video_path,num_frames=self.max_frames)

            processed_frames = [self.vis_processor(image) for image in images]
            num_frames = len(images)

            # support multi-turn instruction tuning
            if isinstance(ann['instruction'],list):
                instructions = ann['instruction']
                outputs = ann['output']
                conversations = []
                for turn_i, instruction in enumerate(instructions):
                    instruction = self.text_processor(instruction)
                    instruction = instruction.replace("<Img><ImageHere></Img>","<Img>"+"<ImageHere>"*num_frames+"</Img>")
                    output = outputs[turn_i]
                    conversations.extend(
                        [
                            {"from": "user", "value":instruction},
                            {"from": "assistant", "value": output},
                        ]
                    )
            else:
                instruction = self.text_processor(ann['instruction'])
                instruction = instruction.replace("<Img><ImageHere></Img>","<Img>"+"<ImageHere>"*num_frames+"</Img>")

                output = ann['output']

                conversations = [
                    {"from": "user", "value":instruction},
                    {"from": "assistant", "value": output},
                ]

            return {
                "image": processed_frames,
                "conversations": conversations,
            }
        except Exception as e:
            print(f"Error loading data at index {index}: {e}")
            return self.__getitem__(random.randint(0,self.__len__()-1))  
