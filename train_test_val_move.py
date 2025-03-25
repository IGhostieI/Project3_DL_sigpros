import os
import shutil
from tqdm import tqdm
import numpy as np

def main():
    source_path = os.path.join(os.getcwd(), "/home/stud/casperc/bhome/Project3_DL_sigpros/generated_data/2025_03_21-21_11_36-standard_amplitude_npz")
    train_path = os.path.join(os.getcwd(),"generated_data/train_2")
    #test_path = "/bhome/casperc/Project1(DeepFit)/generated_data/test"
    val_path = os.path.join(os.getcwd(),"generated_data/val_2")

    os.makedirs(train_path,exist_ok=True)
    #os.makedirs(test_path,exist_ok=True)
    os.makedirs(val_path,exist_ok=True)
    configurations = os.listdir(source_path)
    val_percentage = 20 # %
    for config in configurations:
        print("Configuration: ", config)
        files = os.listdir(os.path.join(source_path, config))
        train_files = files[:int((100-val_percentage)/100*len(files))]
        #test_files = files[int(0.7*len(files)):int(0.85*len(files))]
        val_files = files[int((100-val_percentage)/100*len(files)):]
        print("Train files:", len(train_files))
        #print("Test files:", len(test_files))
        print("Val files:", len(val_files))

        print(f"Moving {config}-files to train folder")
        for file in tqdm(train_files):
            source_file = os.path.join(source_path, config, file)
            destination_file = os.path.join(train_path, file)
            shutil.copy(source_file, destination_file)

        """     
        print("Moving files to test folder")
        for file in tqdm(test_files):
            source_file = os.path.join(source_path, file)
            destination_file = os.path.join(test_path, file)
            shutil.copy(source_file, destination_file) """
            
        print(f"Moving {config}-files to val folder")
        for file in tqdm(val_files):
            source_file = os.path.join(source_path, config, file)
            destination_file = os.path.join(val_path, file)
            shutil.copy(source_file, destination_file) 
if __name__ == "__main__":
    main()  
