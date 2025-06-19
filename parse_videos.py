import os
import re

root = "data"

def parse_videos(root_folders):
    labeled = []
    unlabeled = []
    for root in root_folders:
        for subfolder in os.listdir(root):
            path = os.path.join(root, subfolder)
            category = subfolder[:2]
            if not os.path.isdir(path): continue
            for video in os.listdir(path):
                full_path = os.path.join(path, video)
                match = re.match(r"(HP|SP)\.([a-z]+)\..*\.mov", video, re.IGNORECASE)

                if match:
                    # print('match:', full_path)
                    label = match.group(2).lower()
                    # print((full_path, label))
                    labeled.append((full_path, category, label))
                else:
                    unlabeled.append((full_path, category, ""))
    return labeled, unlabeled

# video_folders = [os.path.join(root, folder) for folder in os.listdir(root) if os.path.isdir(os.path.join(root, folder))]

# labeled, unlabeled = parse_videos(video_folders)
# print('All labeled:',labeled)
# print('All unlabeled:', unlabeled)
# print('Labeled Count:', len(labeled))
# print('Unlabeled Count:', len(unlabeled))