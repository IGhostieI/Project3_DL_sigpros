import os
import shutil
import re

def extract_folder_and_subject(file_path):
    # Extract folder name and file name
    folder_name = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    
    # Extract subject identifier (e.g., S04) from the file name
    subject_match = re.search(r"(S\d+)", file_name)
    subject_id = subject_match.group(1) if subject_match else None

    # Extract folder type (e.g., G*_P, P*_P, or fmrs_pain)
    folder_type = None
    if "fmrs_pain" in file_path:
        folder_type = "fmrs_pain"
    elif re.search(r"P\d+_P", folder_name):
        folder_type = folder_name
    elif re.search(r"G\d+_P", folder_name):
        folder_type = folder_name
    else:
        # Handle cases where folder type is in the file name (e.g., G8_P_S03.table)
        folder_match = re.search(r"(G\d+_P|P\d+_P)", file_name)
        folder_type = folder_match.group(1) if folder_match else None

    # Validate that both folder type and subject ID are present
    if folder_type and subject_id:
        return folder_type, subject_id
    else:
        return None, None

# Define the source and destination directories
source_base_dir = "/home/stud/casperc/bhome/Project3_DL_sigpros/in_vivo_data/Osprey"
destination_base_dir = "/home/stud/casperc/bhome/Project3_DL_sigpros/in_vivo_data/correlation_analysis"

# Walk through the source directory to find .table files
for root, _, files in os.walk(source_base_dir):
    for file in files:
        if file.endswith(".table"):
            # Construct the full path of the file
            source_file_path = os.path.join(root, file)
            
            # Extract folder type and subject ID using the helper function
            folder_type, subject_id = extract_folder_and_subject(source_file_path)
            
            if folder_type and subject_id:
                # Construct the destination folder path
                destination_folder = os.path.join(destination_base_dir, folder_type)
                
                # Ensure the destination folder exists
                os.makedirs(destination_folder, exist_ok=True)
                
                # Rename the file to S**.table (e.g., S01.table, S02.table, etc.)
                new_file_name = f"{subject_id}.table"
                destination_file_path = os.path.join(destination_folder, new_file_name)
                
                # Copy the file to the destination folder with the new name
                shutil.copy(source_file_path, destination_file_path)
                print(f"Copied and renamed: {source_file_path} -> {destination_file_path}")
            else:
                print(f"Skipping file (unable to extract folder/subject): {source_file_path}")