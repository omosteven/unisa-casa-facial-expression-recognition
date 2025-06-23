import os
import re

root = "data"

def parse_videos(root_folders):
    labeled = []
    unlabeled = []
    for root in root_folders:
        for subfolder in os.listdir(root):
            path = os.path.join(root, subfolder)
            sp_or_hp = subfolder[:2]
            if not os.path.isdir(path): continue
            for video in os.listdir(path):
                full_path = os.path.join(path, video)
                match = re.match(r"(HP|SP)\.([a-z]+)\..*\.mov", video, re.IGNORECASE)

                if match:
                    # print('match:', full_path)
                    emotion = match.group(2).lower()
                    # print((full_path, label))
                    labeled.append((full_path, sp_or_hp, emotion))
                else:
                    unlabeled.append((full_path, sp_or_hp, ""))
    return labeled, unlabeled