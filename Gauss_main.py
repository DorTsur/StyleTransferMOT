import os
import torch
import argparse
from PIL import Image
from torch.autograd import Variable
from util import *
import time
from datetime import datetime
import json
import wandb
import torchvision.transforms as transforms


def load_images(contentPath, content_name, stylePath, style_names):
    contentImgPath = os.path.join(contentPath, content_name)
    contentImg = Image.open(contentImgPath).convert('RGB')
    contentImg = transforms.ToTensor()(contentImg)

    styleImgs = []
    for name in style_names:
        styleImgPath = os.path.join(stylePath, name)
        styleImg = Image.open(styleImgPath).convert('RGB')
        styleImg = transforms.ToTensor()(styleImg)
        styleImgs.append(styleImg)

    fineSize =512
    w, h = contentImg.shape[1:]
    if (w > h):
        if (w != fineSize):
            neww = fineSize
            newh = int(h * neww / w)
            contentImg = contentImg.resize((neww, newh))
            styleImgs = [styleImg.resize((neww, newh)) for styleImg in styleImgs]
    else:
        if (h != fineSize):
            newh = fineSize
            neww = int(w * newh / h)
            contentImg = contentImg.resize((neww, newh))
            styleImgs = [styleImg.resize((neww, newh)) for styleImg in styleImgs]

    return contentImg, styleImgs







parser = argparse.ArgumentParser(description='WCT Pytorch')
parser.add_argument('--contentPath',default='images/content',help='path to train')
parser.add_argument('--stylePath',default='images/style',help='path to train')
parser.add_argument('--workers', default=2, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--vgg1', default='models/vgg_normalised_conv1_1.t7', help='Path to the VGG conv1_1')
parser.add_argument('--vgg2', default='models/vgg_normalised_conv2_1.t7', help='Path to the VGG conv2_1')
parser.add_argument('--vgg3', default='models/vgg_normalised_conv3_1.t7', help='Path to the VGG conv3_1')
parser.add_argument('--vgg4', default='models/vgg_normalised_conv4_1.t7', help='Path to the VGG conv4_1')
parser.add_argument('--vgg5', default='models/vgg_normalised_conv5_1.t7', help='Path to the VGG conv5_1')
parser.add_argument('--decoder5', default='models/feature_invertor_conv5_1.t7', help='Path to the decoder5')
parser.add_argument('--decoder4', default='models/feature_invertor_conv4_1.t7', help='Path to the decoder4')
parser.add_argument('--decoder3', default='models/feature_invertor_conv3_1.t7', help='Path to the decoder3')
parser.add_argument('--decoder2', default='models/feature_invertor_conv2_1.t7', help='Path to the decoder2')
parser.add_argument('--decoder1', default='models/feature_invertor_conv1_1.t7', help='Path to the decoder1')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--fineSize', type=int, default=512, help='resize image to fineSize x fineSize,leave it to 0 if not resize')
parser.add_argument('--outf', default='samples/', help='folder to output images')
# parser.add_argument('--alpha', type=float,default=1, help='hyperparameter to blend wct feature and content feature')
parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on.  default is 0")

parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=1000, help="num epochs")
parser.add_argument('--num_styles', type=int, default=1, help="number of considered style images")
parser.add_argument('--eps', type=float, default=1, help="entropic MOT epsilon value")
parser.add_argument('--init_bary', type=str, default='content', help="initialization of the barycenter at each point")
parser.add_argument('--barycenter_lr', type=float, default=0.0005, help="initial lr for the barycenter")
parser.add_argument('--nemot_lr', type=float, default=0.00001, help="initial lr for the NEMOT")
parser.add_argument('--levels', type=str, default='4', help="enc levels string")
parser.add_argument('--barycenter_weights', type=str, default='uniform', help="barycenter_weights")
parser.add_argument('--max_grad_norm', type=float, default=0.01, help="barycenter_weights grad norm")
parser.add_argument('--max_grad_norm_barycenter', type=float, default=1.0, help="barycenter_weights grad norm")
parser.add_argument('--add_bary_noise', type=int, default=1, help="_")
parser.add_argument('--bary_noise_std', type=float, default=0.1, help="_")
parser.add_argument('--seed', type=int, default=10, help="barycenter_weights")
parser.add_argument('--noise_barycenter', type=int, default=1, help="add gaussian noise to initialized barycenter")
parser.add_argument('--bary_method', type=str, default='gauss', help="add gaussian noise to initialized barycenter")  # generative for NEMOT, guassian for Gaussian transfer

parser.add_argument('--using_wandb', type=int, default=0, help='Use Weights & Biases logging')
parser.add_argument('--wandb_project_name', type=str, default='mot_barycenter', help='wandb project name')
parser.add_argument('--wandb_entity', type=str, default='dortsur', help='wandb entity name')

args = parser.parse_args()

try:
    os.makedirs(args.outf)
except OSError:
    pass






#######
#######
#######
#######
#######
#######
#######

imname = 'in3.jpg'
style_names = ['in1_.jpg','in3.jpg']

contentImg, styleImgs = load_images(args.contentPath, imname, args.stylePath, style_names)


# MOT routine:
datetime_now = datetime.now().strftime('%Y%m%d_%H%M%S')
# Define the folder path
args.folderPath = f'results/GaussTransfer/num_styles_{len(style_names)}/content_{imname}/{datetime_now}'
os.makedirs(args.folderPath, exist_ok=True)
print(f'saving to {args.folderPath}')

args_file = os.path.join(args.folderPath, 'args.json')
with open(args_file, 'w') as f:
    json.dump(vars(args), f, indent=4)
print(f"Experiment args saved to {args_file}")




#get content image
mot_b = trainerGauss(args)


levels = [int(char) for char in args.levels]
newImg = contentImg

saveImg = True

if saveImg:
    vutils.save_image(newImg.data.cpu().float(), os.path.join(args.folderPath, f'original.png'))
    for i, img in enumerate(styleImgs):
        vutils.save_image(img.data.cpu().float(), os.path.join(args.folderPath, f'style_{i + 1}_imname_{imname}.png'))
for level in levels:
    with torch.no_grad():  # generate encoded image tensors (out of the computational graph)
        encStyles = [mot_b.enc[level](styleImg) for styleImg in styleImgs]
        encContent = mot_b.enc[level](newImg)
    newImgEnc = mot_b.GaussianOTBarycenter(content=encContent, styles=encStyles, level=level)
    with torch.no_grad():
        newImg = mot_b.dec[level](newImgEnc)

    if saveImg:
        vutils.save_image(newImg.data.cpu().float(),
                          os.path.join(args.folderPath, f'level_{level}_imname_{imname}.png'))
    if args.using_wandb:
        wandb.log({f"mapped image level {level}_imname_{imname}": wandb.Image(newImg.data.cpu().float())})


# def styleTransfer(contentImg, styleImgs, saveImg, args, imname):
#     levels = [int(char) for char in args.levels]
#     newImg = contentImg
#     if saveImg:
#         vutils.save_image(newImg.data.cpu().float(), os.path.join(args.folderPath, f'original.png'))
#         for i, img in enumerate(styleImgs):
#             vutils.save_image(img.data.cpu().float(), os.path.join(args.folderPath, f'style_{i+1}_imname_{imname}.png'))
#     for level in levels:
#         with torch.no_grad():  # generate encoded image tensors (out of the computational graph)
#             encStyles = [mot_b.enc[level](styleImg) for styleImg in styleImgs]
#             encContent = mot_b.enc[level](newImg)
#         if args.bary_method=='generative':
#             newImgEnc = mot_b.solveMOTBarycenterGenerative(content=encContent, styles=encStyles, level=level)
#         elif args.bary_method=='gauss':
#             newImgEnc = mot_b.GaussianOTBarycenter(content=encContent, styles=encStyles, level=level)
#         else:
#             newImgEnc = mot_b.solveMOTBarycenter(content=encContent, styles=encStyles)
#         newImg = mot_b.dec[level](newImgEnc)
#
#
#         if saveImg:
#             vutils.save_image(newImg.data.cpu().float(), os.path.join(args.folderPath,f'level_{level}_imname_{imname}.png'))
#         if args.using_wandb:
#             wandb.log({f"mapped image level {level}_imname_{imname}": wandb.Image(newImg.data.cpu().float())})





# cImg = torch.Tensor()
# sImg = torch.Tensor()
# csF = torch.Tensor()
# csF = Variable(csF)
# if(args.cuda):
#     cImg = cImg.cuda(args.gpu)
#     sImg = sImg.cuda(args.gpu)
#     mot_b.cuda(args.gpu)
# for i,(contentImg,styleImg,imname) in enumerate(loader):
#     # if i > 0:
#     #     break
#     imname = imname[0]
#     if imname != 'in3_.jpg':
#         continue
#     print('MOT Transferring ' + imname)
#     with torch.no_grad():
#         sImg = [styleImg]
#         cImg = contentImg
#
#     start_time = time.time()
#     # WCT Style Transfer
#     styleTransferMOT(cImg, sImg, saveImg=True, args=args, imname=imname)


