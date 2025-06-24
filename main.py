import os
import torch
from parse_videos import parse_videos
from extract_frame import extract_frames
from train_model import TrainModel
import random
import time
class FacialExpressionRecognition:
    def __init__(self):
        self.root_data = 'data'
        self.video_folders = [os.path.join(self.root_data, folder) for folder in os.listdir(self.root_data) if
                         os.path.isdir(os.path.join(self.root_data, folder))]
        self.labeled = []
        self.unlabeled = []
        self.labeled_train = []
        self.labeled_test = []
        self.dataset_file_name = 'processed_labeled_dataset.pt'

    def categorize_videos(self):
        start_time = time.time()
        labeled, unlabeled = parse_videos(self.video_folders)
        self.labeled = labeled
        self.unlabeled = unlabeled
        print("Num of Categorized Videos:", len(self.labeled))
        print("Num of Uncategorized Videos:", len(self.unlabeled))
        elapsed_time = time.time() - start_time
        print(f"Secs taken for the action: {elapsed_time:.4f} seconds")

    def extract_frames_from_categorized_videos(self):
        labeled_frames = []
        start_time = time.time()
        for i, labeled in enumerate(self.labeled):
            frames = extract_frames(labeled[0])
            labeled_frames.append((labeled[0], labeled[1], labeled[2], frames))
        self.labeled = labeled_frames
        elapsed_time = time.time() - start_time
        print(self.labeled)
        print(f"Secs taken for the action: {elapsed_time:.4f} seconds")


    def convert_all_unlabeled_videos_to_frames(self):
        unlabeled_frames = []
        for i, unlabeled in enumerate(self.unlabeled):
            frames = extract_frames(unlabeled[0])
            unlabeled_frames.append((unlabeled[0], unlabeled[1], unlabeled[2], frames))
        self.unlabeled = unlabeled_frames
        print(self.unlabeled)

    def save_preprocessed_videos_as_file(self):
        torch.save(self.labeled, self.dataset_file_name)

    def split_labeled_data_set(self):
        start_time = time.time()
        data = torch.load(self.dataset_file_name)
        random.shuffle(data)
        train_ratio = 0.8
        split_index = int(len(data) * train_ratio)
        self.labeled_train = data[:split_index]
        self.labeled_test = data[split_index:]
        elapsed_time = time.time() - start_time
        print(f"Secs taken for the data split action: {elapsed_time:.4f} seconds")

    def save_train_test_as_file(self):
        start_time = time.time()
        torch.save(self.labeled_train, "train_dataset.pt")
        torch.save(self.labeled_test, "test_dataset.pt")
        print(f"Train size: {len(self.labeled_train)}")
        print(f"Test size: {len(self.labeled_test)}")
        elapsed_time = time.time() - start_time
        print(f"Secs taken for the split saving action: {elapsed_time:.4f} seconds")

    def train_predict_labeled_data_set_with_cnn(self):
        start_time = time.time()
        train_model = TrainModel(train_file="train_dataset.pt", test_file='test_dataset.pt')
        train_model.train_model()

        train_elapsed_time = time.time() - start_time
        print(f"Secs taken for the training action: {train_elapsed_time:.4f} seconds")
        test_start_time = time.time()
        predictions = train_model.make_predictions()
        elapsed_time = time.time() - test_start_time
        print(f"Secs taken for the prediction action: {elapsed_time:.4f} seconds")
        print("The predictions are:", predictions)


facial = FacialExpressionRecognition()
# facial.categorize_videos()
# facial.extract_frames_from_categorized_videos()
# facial.convert_all_unlabeled_videos_to_frames()
# facial.save_preprocessed_videos_as_file()
# facial.split_labeled_data_set()
# facial.save_train_test_as_file()
facial.train_predict_labeled_data_set_with_cnn()
