import random
import shutil
from pathlib import Path
#This splits the dataset. Took random images on each letter to make a test set

def sample_dataset(
    source_dir,
    target_dir,
    images_per_class=100,
    valid_ext={".jpg", ".jpeg", ".png"}
):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    for class_folder in source_dir.iterdir():
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name
        target_class_dir = target_dir / class_name
        target_class_dir.mkdir(exist_ok=True)

        images = [
            img for img in class_folder.iterdir()
            if img.suffix.lower() in valid_ext
        ]

        if len(images) == 0:
            print(f"No images found for {class_name}")
            continue

        # Randomly sample (or take all if fewer than requested)
        selected_images = random.sample(
            images,
            min(images_per_class, len(images))
        )

        for img_path in selected_images:
            shutil.copy(
                img_path,
                target_class_dir / img_path.name
            )

        print(f"{class_name}: copied {len(selected_images)} images")



# sample_dataset(
#     source_dir="C://Users//zylge//Desktop//Datasets//SignAlphaSet",
#     target_dir="C://Users//zylge//Desktop//Datasets//SignAlphaSet_Sampled",
#     images_per_class=200
# )

