import os, shutil

original_img_dir = 'datasets/train/images'
original_lbl_dir = 'datasets/train/labels'

bg_img_dir = 'datasets/train/background/images'
bg_lbl_dir = 'datasets/train/background/labels'

import os, shutil

# Paths
src_img_dir = original_img_dir       # source images
bg_img_dir  = bg_img_dir  # target background images
bg_lbl_dir  = bg_lbl_dir   # target background labels

os.makedirs(bg_img_dir, exist_ok=True)
os.makedirs(bg_lbl_dir, exist_ok=True)

# 2. Get first N images (adjust number if needed)
N = 1000
image_files = sorted([f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.png'))])

if len(image_files) == 0:
    print("⚠️ No images found in the training folder.")
else:
    print(f"Found {len(image_files)} images. Creating {min(N, len(image_files))} background samples...")

    for i, img_file in enumerate(image_files[:N]):
        img_ext = img_file.split('.')[-1]
        new_img_name = f'bg_{i}.{img_ext}'
        new_lbl_name = f'bg_{i}.txt'

        # Copy image
        shutil.copy(os.path.join(src_img_dir, img_file), os.path.join(bg_img_dir, new_img_name))

        # Create empty label
        label_path = os.path.join(bg_lbl_dir, new_lbl_name)
        with open(label_path, 'w') as f:
            pass  # write nothing — this makes it empty

    print("✅ Background images and empty labels created successfully.")