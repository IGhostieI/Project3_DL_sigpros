import os
import shutil

path = "/home/stud/casperc/bhome/Project3_DL_sigpros/in_vivo_data/dat_files"

GROUPS = os.listdir(path)
for group in GROUPS:
    group_path = os.path.join(path, group)
    subjects = os.listdir(group_path)
    for subject in subjects:
        subject_path = os.path.join(group_path, subject)
        if os.path.isdir(subject_path):  # Ensure it's a directory
            files = os.listdir(subject_path)
            for file in files:
                file_path = os.path.join(subject_path, file)
                shutil.move(file_path, group_path)  # Move file to group folder
            os.rmdir(subject_path)

