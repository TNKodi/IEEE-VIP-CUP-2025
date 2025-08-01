import os

label_dir = 'dataset/train/background/labels'  # or wherever your background labels are

for label_file in os.listdir(label_dir):
    label_path = os.path.join(label_dir, label_file)
    with open(label_path, 'w') as f:
        pass  # this clears the file content
