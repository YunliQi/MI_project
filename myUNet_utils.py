import torch
import torchvision 
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from torchvision import transforms 
from torch.utils.data import DataLoader
from PIL import Image # For reading your custom data images
import os
import glob

# Load data and include prepared transform (Remember to apply same transform to both image and label) 
class mySegmentationData(object):
    def __init__(self, root, transforms = None):
        self.root = root
        self._eval = eval
        self.transforms = transforms
        self.build_dataset()
        
    def build_dataset(self):   
        self.imgs = os.path.join(self.root, "nnUNet_data", "Dataset001_ISIC2018", "imagesTr")
        self.masks = os.path.join(self.root, "nnUNet_data", "Dataset001_ISIC2018", "labelsTr_255")
        
        self._images = sorted(glob.glob(self.imgs + "/*.png"))
        
#       Note: Here we are using only left lung data today
        self._labels = sorted(glob.glob(self.masks  + "/*.png"))
        
    def __getitem__(self, idx):
    
        img = Image.open(self._images[idx]).convert("RGB").resize((256,256), resample=0)
        mask = Image.open(self._labels[idx]).convert("L").resize((256, 256), resample=0)
        mask = Image.fromarray(((np.asarray(mask)).astype('float32')).astype('uint8'))
        if self.transforms is not None:
            img = self.transforms(img)
            mask = self.transforms(mask)
   
        return img, mask
    
    def __len__(self):
        return len(self._images)
    

# Step 1: Build a nn.Sequential with two conv-BatchN-Relu layers
def baseblock(channel_in, channel_out):
#     Design a baseblock with conv-batch-relu x 2 (each input is twice convolved as in fig.)
    return nn.Sequential(
        nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(inplace=True)
    )  

# Step 2: Build a downscaling module [Hint: use the above layeredConv after that]
# Add a maxpool before baseblock as in figure
def downsamplePart(channel_in, channel_out):
    return nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            baseblock(channel_in, channel_out)
        )

# Step 3: Build a upscaling module [Hint: use the above layeredConv after that]
# - Remember there is also concatenation and size may change so we are padding
class upsampledPart(nn.Module):
    def __init__(self, channel_in, channel_out, bilinear=True):
        super().__init__()
        #self.up = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)
        self.up = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=2, stride=2)
        self.conv = baseblock(channel_in, channel_out)
        
    def forward(self, x1, x2):
        # upscale and then pad to eliminate any difference between upscaled and other feature map coming with skip connection     
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2] )
        
        # concatenate (perform concatenation of x1 and x2 --> remember these are skip x2(from encoder) and upssampled image x1)
        x = torch.cat([x2, x1], dim=1)
        
        # apply baseblock after concatenation --> you do again two convs.? --> baseblock
        x = self.conv(x)

        return x
    
# Step 4: Compile all of above together
# here output channel should be equal to number of classes
class UNet(nn.Module):
    def __init__(self, channel_in, channel_out, bilinear=None):
        super(UNet,self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        
        #call your base block
        self.initial = baseblock(channel_in, 64)
        
        # downsampling layers with 2 conv layers
        self.down1 = downsamplePart(64, 128)
        self.down2 = downsamplePart(128, 256)
        self.down3 = downsamplePart(256, 512)
        self.down4 = downsamplePart(512, 1024)
        
        # your code here
        # upsampling layers with feature concatenation and 2 conv layers 
        self.up1 = upsampledPart(1024, 512) 
        self.up2 = upsampledPart(512, 256)
        self.up3 = upsampledPart(256, 128)
        self.up4 = upsampledPart(128, 64)
        
        # output layer
        self.out = nn.Conv2d(64, channel_out, kernel_size=1) 

    # build a forward pass here
    # remember to keep your output as you will need to concatenate later in upscaling
    def forward(self,x):
        x1 = self.initial(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # your code here for upscaling
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # output
        return self.out(x)  


# Create metrics
def dice(a, b):
    """Calculate dice score for each image in tensor"""
    # a and b are tensors of shape (B, C, H, W)
    # Sum over last two axes (H and W i.e. each image)
    return 2*(a*b).sum(axis=[-2, -1])/(a + b).sum(axis=[-2,-1]).type(torch.float32)

def mask_out(out):
    """Mask tensor/array with 0 threshold"""
    # Need to binarize the output to be able to calculate dice score
    return out > 0

def get_dice_arr(out, label):
    """Get dice score for each image in the batch for each mask seperately"""
    # Output is shape (B, C)
    return dice(mask_out(out), label)