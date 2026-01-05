from pathlib import Path

#Pre processing - adding keypoints to ASL images

path = Path("C://Users//zylge//Downloads//asl_dataset") # get the path of the dataset.

#print(path)

#loops through all files and subfolders insided tha path and returns the full path of each items
for items in path.iterdir():
    image_folder_path = items

    #loops through every image or subfolder of the current subfolder and returns the full path of each items
    for image_items in image_folder_path.iterdir():
        image_paths = image_items






