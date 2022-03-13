import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import torch
import numpy as np
import segmentation_models_pytorch as smp
from pytorch_toolbelt import losses as L
import torch.nn as nn
import argparse


#Is Pytoch using a gpu?
print("Cuda available?", torch.cuda.is_available())

# Which GPU Is The Current GPU?
print("GPU:", torch.cuda.current_device())

############################################################################
# helper functions for data visualization
def visualize(image,mask, classes, idx):
    ##PLot images in one row.
    n = len(classes)
    fig = plt.figure(figsize=(16, 5))
    if len(classes) != 0:
        for i, class_name in enumerate(classes):
            if 1 in mask[i,:,:]:
                print(f"{class_name} is in the mask for {idx}")
                plt.xticks([])
                plt.yticks([])
                plt.subplot(1,2,1)
                plt.imshow(image)
                plt.subplot(1,2,2)
                plt.imshow(mask[i,:,:].squeeze())
                fig.savefig(f"fig_results/{idx}_{class_name}.png")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(mask[-1,:,:].squeeze())
    fig.savefig(f"fig_results/{idx}_background.png")
    #plt.show()

def visualize_rgbmask(image,mask, idx):
    ##PLot images in one row.
    fig = plt.figure(figsize=(16, 5))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(mask)
    #plt.imshow(mask.squeeze())
    fig.savefig(f"fig_results/{idx}.png")

###############################################################################
# Define the class of dataset
class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    #with open(objects_names) as f:
    #    objects = f.readlines()
    #CLASSESALL = [line[:-1] for line in objects]
    #print(CLASSES)
    #with open(objects_colors) as f:
    #    colors = f.readlines()
    #COLORS = [line[:-1] for line in colors]
    #print(COLORS)


    def __init__(
            self, 
            data_dir, 
            list_data, 
            classes=None, 
            augmentation=None, 
            preprocessing=None, colors= None):
        ### classes that we are looking for to get segmetation results
        self.classes = classes 
        data = pd.read_csv(list_data, sep=" ", header = None)
        data.columns = ['image','mask']
        #print(list(data['mask'].to_numpy()))
        self.ids = list(data['image'].to_numpy())
        self.mask_ids = list(data['mask'].to_numpy())
        #print(self.ids)
        self.images_fps = [os.path.join(data_dir, image_id[1:]) for image_id in self.ids]
        #print(self.images_fps)
        self.masks_fps = [os.path.join(data_dir, mask_id[1:]) for mask_id in self.mask_ids]
        #print(len(self.masks_fps))
        # convert str names to class values on masks
        self.class_values = [self.classes.index(cls) for cls in classes]
        #print(classes)
        #print(self.class_values)   
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.colors = colors
    
    def __getitem__(self, i):   
        # read data
        #print(i)
        #print(self.images_fps[i])
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i])
        #print(mask)
        #print(list(mask[2,2,:]))    
        # extract certain classes from mask (e.g. cars)
        colors_of_classes_str = [self.colors[class_id] for class_id in self.class_values]
        #print(colors_of_classes_str)
        colors_of_classes_int=[]
        for color_class in colors_of_classes_str:
        	color_class_int = [int(i) for i in color_class.split()]
        	colors_of_classes_int.append([color_class_int[2],color_class_int[1],color_class_int[0]])
        #print(colors_of_classes_int)
        #print(mask)

        masks = [torch.eq(torch.from_numpy(mask),torch.tensor(v, dtype=torch.int)).numpy() for v in colors_of_classes_int]
        #print(masks)
        masks = [torch.min((torch.tensor(mask)).long(),2)[0] for mask in masks]
        mask = np.stack(masks, axis=-1).astype('float')   
        #print(mask.shape)
        # add background if mask is not binary
        #if mask.shape[-1] != 1:
        #    background = 1 - mask.sum(axis=-1, keepdims=True)
        #    mask = np.concatenate((mask, background), axis=-1)
        #print(mask[:,:,8])

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask     

    def __len__(self):
        return len(self.ids)

# Lets look at data we have
#selected_classes = ['Bed', 'Laptop', 'Book', 'Pillow', 'SideTable', 'Desk', 'Pen']
#selected_classes = ['Bed', 'Book', 'Drawer', 'Shelf', 'Cabinet', 'SideTable', 'Desk', 'AlarmClock']
#weight_CE = torch.Tensor([2,20,2,10,2,10,10,20,1]).cuda()
#selected_classes = ['Bed', 'Book', 'Drawer']
#print(selected_classes.append("Background"))
#print(selected_classes.append("Background"))

# ================================= Computing weights (inverse of numbers of objects) =====================================
"""
dic_object_count = {key: 0 for key in CLASSESALL}
dic_object_pixels = {key: 0 for key in CLASSESALL}
#print(dic_object_count)
sum_of_pixels_background = 0
for i in range(6000):
    print(i)
    image, mask = dataset[i]
    for j, object_n in enumerate(CLASSESALL):
        if 1 in mask[:,:,j]:
            #print("t")
            dic_object_count[object_n]+=1
            dic_object_pixels[object_n]+=(mask[:,:,j] == 1).sum()
            sum_of_pixels_background += (mask[:,:,-1] == 1).sum()
        #print("count =", dic_object_count)
        #print("pixels =", dic_object_pixels)
objects_weights = [1/dic_object_pixels[class_n] if dic_object_pixels[class_n]!=0 else 0 for class_n in CLASSESALL] ### if you want to consider weight based on inverse of pixels of objects in the images
#objects_weights = [1/dic_object_count[class_n] if dic_object_count[class_n]!=0 else 0 for class_n in CLASSESALL] ### if you want to consider weight based on inverse of numbers of objects in the images
#print(dic_object_count)
#print("weights for objects =", objects_weights)
#print("background_pixels =", 1/sum_of_pixels_background)
weights_file = open("segmentation_data_alfred/alfred_classes_file/weights_inverse_numbers_of_obj_&_their_avg_pixels.txt","w")
for obj_w in objects_weights:
    weights_file.write(str(obj_w*300*300)+" \n")
weights_file.write(str((1/sum_of_pixels_background)*300*300)+" \n")
weights_file.close()
"""
# =========================================================================================================================
#image, mask = dataset[7] # get some sample
#print(image.shape)
#print(mask.shape)
#mask = np.transpose(mask,(2,0,1))
#print(mask.shape)
#visualize(
#    idx = 1,
#    image=image, 
#    mask=mask,
#    classes=selected_classes
#)

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=288, min_width=288, always_apply=True, border_mode=0),
        albu.RandomCrop(height=288, width=288, always_apply=True),

        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    #Add paddings to make image shape divisible by 32
    test_transform = [
        albu.PadIfNeeded(320, 320)
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    #Construct preprocessing transform
    
    """
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

# same image with different random transforms
#for i in range(3):
    #print('True')
    #image, mask = augmented_dataset[7]
    #visualize(idx=i, image=image, mask=mask, classes=selected_classes)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--data_path',type=str,default= 'segmentation_data_alfred/alfred/')
    parser.add_argument('--data_file_dir',type=str,default= 'segmentation_data_alfred/alfred_classes_file/')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint')
    parser.add_argument('--test_checkpoint_path', type=str, default='./checkpoint/deeplabv3/deeplabv3_all/best_model_266.pth')
    parser.add_argument('--fig_results_dir', type=str, default='fig_results')
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--train', action="store_true", default = False)
    parser.add_argument('--test', action="store_true", default = True)
    parser.add_argument('--test_on_valid', action="store_true", default = False)
    args = parser.parse_args()

    ###########################################################################
    # loading data
    list_train = os.path.join(args.data_path, 'list/training.txt')
    list_validation = os.path.join(args.data_path, 'list/validation.txt')
    list_test = os.path.join(args.data_path, 'list/test.txt')
    objects_names = os.path.join(args.data_file_dir, 'alfred_names.txt')
    objects_colors = os.path.join(args.data_file_dir, 'alfred_colors.txt')
    objects_weights = os.path.join(args.data_file_dir, 'weights_inverse_numbers_of_obj_&_their_avg_pixels.txt')

    with open(objects_names) as f:
        objects = f.readlines()
    CLASSESALL = [line[:-1] for line in objects]
    #print(CLASSES)
    with open(objects_colors) as f:
        colors = f.readlines()
    COLORS = [line[:-1] for line in colors]
    #print(COLORS)

    with open(objects_weights) as f: ## added a weight at the end of the file for background
        weights = f.readlines()
    weightsall = [line[:-1] for line in weights]
    weightsall = [float(i) for i in weightsall] 
    #print(len(weightsall))

    selected_classes = CLASSESALL
    dataset = Dataset(args.data_path, list_train, classes= selected_classes, colors= COLORS)

    augmented_dataset = Dataset(
    args.data_path,
    list_train, 
    augmentation=get_training_augmentation(), 
    classes=selected_classes, colors= COLORS)

    ##########################################################################################
    #### Create Model and Train ####

    ENCODER = 'se_resnext50_32x4d'
    #ENCODER = 'efficientnet-b4'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = selected_classes
    ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'
    n_classes = len(CLASSES)

    # create segmentation model with pretrained encoder

    ### FPN and UNET++ models
    #model = smp.FPN(
    #model = smp.UnetPlusPlus(
    #    encoder_name=ENCODER, 
    #    encoder_weights=ENCODER_WEIGHTS, 
    #    classes=n_classes, 
    #    activation=ACTIVATION,
    #)

    ### DeepLabv3+ model
    #model = smp.DeepLabV3Plus(
    #    encoder_name=ENCODER, 
    #    encoder_weights=ENCODER_WEIGHTS, 
    #    classes=n_classes, 
    #    activation=ACTIVATION,
    #)

    ### DeepLab V3 model
    model = smp.DeepLabV3(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=n_classes, 
        activation=ACTIVATION,
    )


    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    loss = smp.utils.losses.DiceLoss() 

    #loss = smp.losses.SoftBCEWithLogitsLoss(pos_weight = w)
    #loss.__name__ = "BCELoss"

    ### Weighted Cross Entropy
    #weight_CE = torch.Tensor(weightsall).cuda()
    #loss = nn.CrossEntropyLoss(weight=weight_CE)
    #loss.__name__ = "weightedCELoss"

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])


    ### Loading data
    valid_dataset = Dataset(
            args.data_path,
            list_validation, 
            augmentation=get_validation_augmentation(), 
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=selected_classes,
            colors= COLORS
        )
    valid_dataset_vis = Dataset(
            args.data_path, 
            list_validation,
            classes=CLASSES,
            colors= COLORS
        )
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    valid_epoch = smp.utils.train.ValidEpoch(
            model, 
            loss=loss, 
            metrics=metrics, 
            device=DEVICE,
            verbose=True,
        )

    if args.train:
        train_dataset = Dataset(
            args.data_path,
            list_train,
            augmentation=get_training_augmentation(), 
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=selected_classes,
            colors= COLORS
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=14)
        # create epoch runners 
        # it is a simple loop of iterating over dataloader`s samples
        train_epoch = smp.utils.train.TrainEpoch(
            model, 
            loss=loss, 
            metrics=metrics, 
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        )

        #image_train, gt_mask_train = train_dataset[5]
        #print("image and mask shape of train =", image_train.shape, gt_mask_train.shape)
        #for batch, sample in enumerate(train_loader):
        #    print("Batch =", batch)
        #    print("X_Sample =", sample[0].shape) x:(8,3,288,288) y:(8,9,288,288)
        #    print("Y_Sample =", sample[1].shape)

    ### Training loop
        max_score = 0
        #train accurascy, train loss, val_accuracy, val_loss をグラフ化できるように設定．
        x_epoch_data = []
        train_dice_loss = []
        train_iou_score = []
        valid_dice_loss = []
        valid_iou_score = []
        for i in range(0, args.epoch):  
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            print(train_logs)
            valid_logs = valid_epoch.run(valid_loader)
            print(valid_logs)

            x_epoch_data.append(i)
            train_dice_loss.append(train_logs['weightedCELoss'])
            train_iou_score.append(train_logs['iou_score'])
            valid_dice_loss.append(valid_logs['weightedCELoss'])
            valid_iou_score.append(valid_logs['iou_score'])

            #=================================================================================
            ### Debugging purpose ###
            """for n in range(0,10):
                n = np.random.choice(len(valid_dataset))
                image_valid, gt_mask_valid = valid_dataset[n]    
                image_vis_valid = valid_dataset_vis[n][0].astype('uint8')
                print("image and mask shape of valid =", image_valid.shape, gt_mask_valid.shape)
                #print("size gt mask valid =", gt_mask_valid.shape)
                print("is there any 1 in gorund truth mask =", 1 in gt_mask_valid[:-1,:,:])
                if 1 in gt_mask_valid[:-1,:,:]:
                    visualize(idx=f'deeplabv3+_background_weighted2/{n}_GROUNDTRUTH_epoch_{i}', image=image_vis_valid, mask=gt_mask_valid, classes = selected_classes)
                    x_tensor = torch.from_numpy(image_valid).to(DEVICE).unsqueeze(0)
                    pr_mask_epoch = model.predict(x_tensor)
                    #pr_mask_epoch = torch.mul(pr_mask_epoch,1e6)
                    #print(pr_mask_epoch)
                    pr_mask_epoch = (pr_mask_epoch.squeeze().cpu().numpy().round())
                    #print(pr_mask_epoch[-1,:,:])
                    print("is there any 1 in predicted mask except background=", 1 in pr_mask_epoch[:-1,:,:])
                    #print("shape_mask_pr =", pr_mask_epoch.shape)
                    #print(pr_mask_epoch)
                    if 1 in pr_mask_epoch[:-1,:,:]:
                        visualize(idx=f'deeplabv3+_background_weighted2/{n}_PREDICTION_epoch_{i}', image=image_vis_valid, mask=pr_mask_epoch, classes = selected_classes)
            ### Debugging purpose ###"""
            #===================================================================================

            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(model, args.checkpoint_path + f'/epoch_{i}.pth')
                print('Model saved!')            
            if i == 25:
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')
            if i == 50:
                optimizer.param_groups[0]['lr'] = 5e-6
                print('Decrease decoder learning rate to 5e-6!')
            
            if i == 75:
                optimizer.param_groups[0]['lr'] = 1e-6
                print('Decrease decoder learning rate to 1e-6!')

    #######################################################################################
    #Test a trined checkpoint
    if args.test:
        # load best saved checkpoint
        best_model = torch.load(args.test_checkpoint_path)
        # create test datase
        test_dataset = Dataset(
            args.data_path,
            list_test, 
            augmentation=get_validation_augmentation(), 
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=selected_classes,
            colors= COLORS
        )
        test_dataloader = DataLoader(test_dataset)
        # evaluate model on test set
        test_epoch = smp.utils.train.ValidEpoch(
            model=best_model,
            loss=loss,
            metrics=metrics,
            device=DEVICE,
        )
        logs = test_epoch.run(test_dataloader)
        print(logs)
        # test dataset without transformations for image visualization
        test_dataset_vis = Dataset(
            args.data_path,
            list_test, 
            classes=selected_classes,
            colors= COLORS
        )

        ### validation dataset ###
        for i in range(20):
            if args.test_on_valid:
                #n = 9
                n = np.random.choice(len(valid_dataset))
                image_vis = valid_dataset_vis[n][0].astype('uint8')
                image, gt_mask = valid_dataset[n]
            else:
                n = np.random.choice(len(test_dataset))
                #n = 10
                image_vis = test_dataset_vis[n][0].astype('uint8')
                image, gt_mask = test_dataset[n]
            #print(image.shape)
            gt_mask = gt_mask.squeeze() 
            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            #print(torch.min(x_tensor), torch.max(x_tensor))
            #print(x_tensor)
            pr_mask = best_model.predict(x_tensor)
            #print(pr_mask)
            #print(pr_mask.shape)
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())    
            #print(pr_mask[-1,:,:]) 
            #print(gt_mask[-1,:,:]) 
            #print(np.expand_dims(pr_mask[-1,:,:], axis=0))
            #visualize(
            #    image=image_vis, 
            #    ground_truth_mask=gt_mask, 
            #    predicted_mask=pr_mask
            #)
            #print(gt_mask.shape)
            if 1 in gt_mask:

                pr_rgb_mask = torch.zeros(320,320,3)
                for i in range(len(selected_classes)):

                    indices = (pr_mask[i,:,:] == 1).nonzero()
                    for j in range(len(indices[0])):

                        idx_obj_img = [indices[0][j],indices[1][j]]

                        class_alfred = selected_classes[i]
                        #print(class_alfred)
                        idx_object = CLASSESALL.index(class_alfred)
                        color_obj = COLORS[idx_object]
                        color_class = [int(i) for i in color_obj.split()]
                        pr_rgb_mask[idx_obj_img[0],idx_obj_img[1]] = torch.tensor(color_class)

                #plt.imsave('mask_pr.png', rgb_mask.numpy().astype('uint8'))
                pr_rgb_mask = pr_rgb_mask.numpy().astype('uint8')


                gt_rgb_mask = torch.zeros(320,320,3)
                for i in range(len(selected_classes)):

                    indices = (gt_mask[i,:,:] == 1).nonzero()
                    for j in range(len(indices[0])):

                        idx_obj_img = [indices[0][j],indices[1][j]]

                        class_alfred = selected_classes[i]
                        #print(class_alfred)
                        idx_object = CLASSESALL.index(class_alfred)
                        color_obj = COLORS[idx_object]
                        color_class = [int(i) for i in color_obj.split()]
                        gt_rgb_mask[idx_obj_img[0],idx_obj_img[1]] = torch.tensor(color_class)

                #plt.imsave('mask_pr.png', rgb_mask.numpy().astype('uint8'))
                gt_rgb_mask = gt_rgb_mask.numpy().astype('uint8')
                #rgb_mask = np.transpose(rgb_mask, (2, 0, 1))
                #visualize(
                #    idx=f'UNET++background/{n}_gt',
                #    image=image_vis, 
                #    mask=np.expand_dims(gt_mask[-1,:,:], axis=0),
                #    classes=[])
                #visualize(
                #    idx=f'deeplabv3/epoch_266/{n}_gt_test',
                #    image=image_vis, 
                #    mask=gt_mask,
                #    classes=selected_classes)
                #visualize(
                #    idx = f'deeplabv3/epoch_266/{n}_pr_test',
                #    image=image_vis, 
                #    mask=pr_mask,
                #    classes=selected_classes)
                #visualize_rgbmask(
                #    idx=f'deeplabv3/epoch_266/{n}_rgb_mask',
                #    image=image_vis, 
                #    mask=pr_rgb_mask)
                plt.close('all')
                fig = plt.figure(figsize=(5, 5),frameon=False)
                plt.imshow(image_vis)
                plt.xticks([])
                plt.yticks([])
                fig.savefig(args.fig_results_dir +f"/{n}_image_single.png")

                fig = plt.figure(figsize=(5, 5),frameon=False)
                plt.imshow(pr_rgb_mask)
                plt.xticks([])
                plt.yticks([])
                fig.savefig(args.fig_results_dir +f"/{n}_pr_rgb_mask_single.png")

                fig = plt.figure(figsize=(5, 5),frameon=False)
                plt.imshow(gt_rgb_mask)
                plt.xticks([])
                plt.yticks([])
                fig.savefig(args.fig_results_dir + f"/{n}_gt_rgb_mask_single.png")