import os
import shutil

# ===== CHANGE ONLY IF YOUR USERNAME IS DIFFERENT =====
SOURCE = r"C:\Users\sanja\Downloads\PCB_DATASET\PCB_DATASET\images"

TEMPLATE = "template"
TEST = "test"

os.makedirs(TEMPLATE, exist_ok=True)
os.makedirs(TEST, exist_ok=True)

print("Reading dataset from:", SOURCE)

# check dataset exists
if not os.path.exists(SOURCE):
    print("ERROR: Dataset path not found!")
    exit()

folders = os.listdir(SOURCE)
print("Folders found:", folders)

count = 0

for defect_type in folders:
    folder = os.path.join(SOURCE, defect_type)

    if not os.path.isdir(folder):
        continue

    images = os.listdir(folder)

    if len(images) < 2:
        continue

    print(f"\nProcessing class: {defect_type} ({len(images)} images)")

    # first image → template
    template_img = images[0]
    shutil.copy(
        os.path.join(folder, template_img),
        os.path.join(TEMPLATE, f"{count:04d}.jpg")
    )

    # remaining images → test images
    for img in images[1:]:
        shutil.copy(
            os.path.join(folder, img),
            os.path.join(TEST, f"{count:04d}.jpg")
        )
        print(f"Pair created → {count:04d}.jpg")
        count += 1

print("\nDataset prepared successfully!")
print("Total pairs created:", count)
