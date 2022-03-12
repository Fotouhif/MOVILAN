from models import UnetAdaptiveBins
import model_io
from PIL import Image
from infer import InferenceHelper
import utils
import torch
import numpy as np

#transform = ToTensor()
#MIN_DEPTH = 1e-3
#MAX_DEPTH_NYU = 10
#MAX_DEPTH_KITTI = 80

#N_BINS = 256 

# NYU
#model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_NYU)
#pretrained_path = "/home/fotouhif/Documents/AdaBins_depth/pretrained_model/AdaBins_nyu.pt"
#model, _, _ = model_io.load_checkpoint(pretrained_path, model)

infer_helper = InferenceHelper(dataset='nyu')

# predict depth of a single pillow image
#img = Image.open("test_imgs/classroom__rgb_00283.jpg")  # any rgb pillow image
#bin_centers, predicted_depth= infer_helper.predict_pil(img)

#viz = utils.colorize(torch.from_numpy(predicted_depth).unsqueeze(0), vmin=None, vmax=None, cmap='magma')
#viz = Image.fromarray(viz)


#image_nyu = np.asarray(Image.open("/home/fotouhif/Documents/AdaBins_depth/test_imgs/classroom__rgb_00283.jpg"), dtype='float32') / 255.
#print(image_nyu.shape)
#image_nyu = transform(image_nyu).unsqueeze(0).to(self.device)

# predict depths of images stored in a directory and store the predictions in 16-bit format in a given separate dir
infer_helper.predict_dir_model_nyu("/home/fotouhif/Documents/MOVILAN/RGB_depth/results/RGB/", "/home/fotouhif/Documents/MOVILAN/RGB_depth/results/Depth_AdaBins1/")
#infer_helper.predict_dir("/home/fotouhif/Documents/AdaBins_depth/test_imgs/", "/home/fotouhif/Documents/AdaBins_depth/save_dir/16bit/") #torch.Size([1, 3, 480, 640])