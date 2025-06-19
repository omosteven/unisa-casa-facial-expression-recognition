import os
import torch
from parse_videos import parse_videos
from extract_frame import extract_frames

class FacialExpressionRecognition:
    def __init__(self):
        self.root_data = 'data'
        self.video_folders = [os.path.join(self.root_data, folder) for folder in os.listdir(self.root_data) if
                         os.path.isdir(os.path.join(self.root_data, folder))]
        labeled, unlabeled = parse_videos(self.video_folders)
        self.labeled = labeled
        self.unlabeled = unlabeled

        first_image = self.unlabeled[0]
        # print(first_image[0])
        # print(len(extract_frames("data/video tesi/HP2/HP.soddisfatto.02.mov")))
        # print(len(extract_frames(first_image[0])))


    def convert_all_labeled_videos_to_frames(self):
        labeled_frames = []
        for i, labeled in enumerate(self.labeled):
            frames = extract_frames(labeled[0])
            labeled_frames.append((labeled[0], labeled[1], labeled[2], frames))
        self.labeled = labeled_frames
        print(self.labeled)

    def convert_all_unlabeled_videos_to_frames(self):
        unlabeled_frames = []
        for i, unlabeled in enumerate(self.unlabeled):
            frames = extract_frames(unlabeled[0])
            unlabeled_frames.append((unlabeled[0], unlabeled[1], unlabeled[2], frames))
        self.unlabeled = unlabeled_frames
        print(self.unlabeled)

    def save_preprocessed_videos_as_file(self):
        torch.save(self.labeled, "processed_labeled_dataset.pt")


facial = FacialExpressionRecognition()
facial.convert_all_labeled_videos_to_frames()
facial.save_preprocessed_videos_as_file()
