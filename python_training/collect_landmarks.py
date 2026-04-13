import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
SEQUENCE_LENGTH = 30  # frames per window

# Landmark indices we care about (from MediaPipe's 33 landmarks)
KEYPOINTS = [
    0,   # nose
    11, 12,  # shoulders
    13, 14,  # elbows
    15, 16,  # wrists
    23, 24,  # hips
    25, 26,  # knees
    27, 28,  # ankles
    29, 30,  # heels
    31, 32   # foot index
]

def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    sequences = []
    frame_buffer = []

    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=2
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Extract x, y, z, visibility for selected keypoints
                frame_data = []
                for idx in KEYPOINTS:
                    lm = landmarks[idx]
                    frame_data.extend([lm.x, lm.y, lm.z, lm.visibility])
                frame_buffer.append(frame_data)

                # Sliding window with 50% overlap
                if len(frame_buffer) == SEQUENCE_LENGTH:
                    sequences.append(np.array(frame_buffer))
                    frame_buffer = frame_buffer[15:]  # 15-frame stride

    cap.release()
    return np.array(sequences)  # shape: (N, 30, 76)

def process_all_videos(video_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_sequences = []


    for fname in os.listdir(video_dir):
        if fname.endswith(('.mp4', '.mov', '.avi')):
            path = os.path.join(video_dir, fname)
            print(f"Processing: {fname}")
            seqs = extract_landmarks(path)

            if len(seqs) > 0:
                all_sequences.append(seqs)
                print(f"  → {len(seqs)} sequences extracted")
            else:
                print(f"  → 0 sequences ❌ (skipped)")

    all_sequences = np.concatenate(all_sequences, axis=0)
    np.save(os.path.join(output_dir, 'good_pushup_sequences.npy'), all_sequences)
    print(f"\nTotal sequences: {len(all_sequences)}, shape: {all_sequences.shape}")
    return all_sequences

if __name__ == '__main__':
    process_all_videos('data/raw_videos', 'data/landmarks')