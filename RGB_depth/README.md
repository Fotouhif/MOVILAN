Link to Alfred depth dataset and checkpoint:

https://drive.google.com/drive/folders/1kA9tl7qBCEjWHwwBSrICFlhnSbdKwnc2?usp=sharing

For training you can download the data folder and put in in this "RGB_depth" folder.
For Using the trained model you need to download checkpint folder and put it in this "RGB_depth" folder.

####
Link to Adabins github:

https://github.com/shariqfarooq123/AdaBins

####
Steps for training on Alfred data:
$wandb login
Then go to the shown website and copy and paste the string
$ python train.py --data_path /home/fotouhif/Documents/MOVILAN/RGB_depth/data/Training_testing/RGB/ --gt_path /home/fotouhif/Documents/MOVILAN/RGB_depth/data/Training_testing/depth/ --filenames_file /home/fotouhif/Documents/MOVILAN/RGB_depth/data/alfred_rgb_depth_train_shuffle.txt --input_height 300 --input_width 300 --min_depth 1 --max_depth 10000 --data_path_eval /home/fotouhif/Documents/MOVILAN/RGB_depth/data/Training_testing/RGB/ --gt_path_eval /home/fotouhif/Documents/MOVILAN/RGB_depth/data/Training_testing/depth/ --filenames_file_eval /home/fotouhif/Documents/MOVILAN/RGB_depth/data/alfred_rgb_depth_test.txt --min_depth_eval 1 --max_depth_eval 10000