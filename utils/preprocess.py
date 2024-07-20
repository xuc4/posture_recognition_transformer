import os
import cv2
import numpy as np
import pickle
import scipy.io

DATASET_PATH = 'dataset/Penn_Action'
OUTPUT_PATH = 'data/processed_data.pkl'

def load_frames_from_folder(folder_path):
    frames = []
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        frame = cv2.imread(image_path)
        frames.append(frame)
    return frames

def load_annotations(mat_path):
    annotations = scipy.io.loadmat(mat_path)
    return annotations

def extract_keypoinnts_and_labels(frames, annotations):
    keypoints = annotations['keypoints']
    labels = annotations['action']
    return keypoints, labels

def preprocess_data():
    data = []
    video_folders = [f for f in os.listdir(os.path.join(DATASET_PATH, 'frames')) if os.path.isdir(os.path.join(DATASET_PATH, 'frames', f))]

    for video_folder in video_folders:
        frames_folder_path = os.path.join(DATASET_PATH, 'frames', video_folder)
        annotations_path = os.path.join(DATASET_PATH, 'labels', f"{video_folder}.mat")

        frames = load_frames_from_folder(frames_folder_path)
        annotations = load_annotations(annotations_path)

        keypoints, labels = extract_keypoinnts_and_labels(frames, annotations)

        data.append({'video': video_folder, 'keypoints': keypoints, 'labels': labels})

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    preprocess_data()
