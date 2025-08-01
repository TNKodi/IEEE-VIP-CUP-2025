import os, random, shutil
from tqdm import tqdm   # remove this line if you didn't install tqdm

# 1. SETTINGS — edit these:
SRC_DIR     = 'Dataset/ALL'      # folder containing both .jpg/.png and .txt files
TRAIN_RATIO = 0.7           # 70% train
VAL_RATIO   = 0.2           # 20% val
# (test will be the remaining 10%)

# 2. Prepare destination folders
for split in ('train','val','test'):
    os.makedirs(os.path.join(SRC_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(SRC_DIR, split, 'labels'), exist_ok=True)

# 3. List all image files
all_files = os.listdir(SRC_DIR)
imgs = [f for f in all_files if f.lower().endswith(('.jpg','.png'))]
random.seed(42)
random.shuffle(imgs)

# 4. Compute split sizes
n = len(imgs)
n_train = int(TRAIN_RATIO * n)
n_val   = int(VAL_RATIO   * n)

train_imgs = imgs[:n_train]
val_imgs   = imgs[n_train:n_train + n_val]
test_imgs  = imgs[n_train + n_val:]

# 5. Move each image + its .txt into the right folder
def move_subset(subset, names):
    for img in (tqdm(names, desc=subset) if 'tqdm' in globals() else names):
        lbl = os.path.splitext(img)[0] + '.txt'
        # source paths
        img_src = os.path.join(SRC_DIR, img)
        lbl_src = os.path.join(SRC_DIR, lbl)
        # dest paths
        img_dst = os.path.join(SRC_DIR, subset, 'images', img)
        lbl_dst = os.path.join(SRC_DIR, subset, 'labels', lbl)
        # move (or use copy if you prefer to keep originals)
        shutil.move(img_src, img_dst)
        shutil.move(lbl_src, lbl_dst)

move_subset('train', train_imgs)
move_subset('val',   val_imgs)
move_subset('test',  test_imgs)

print(f"Done!  Train: {len(train_imgs)},  Val: {len(val_imgs)},  Test: {len(test_imgs)}")
