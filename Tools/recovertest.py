from PIL import Image, ImageDraw, ImageChops
import torch
from torchvision import datasets, models, transforms

def imshow(tensor, imsize=512, title=None):
    image = tensor.clone().cpu()
    image = image.view(*tensor.size())
    image = transforms.ToPILImage()(image).convert('L')
    return image

image_folder = "blocks/output4"
starting_index = 12
starting_stage = starting_index +1

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
    ]
)

# read normalized delta state files
delta_s = torch.load(image_folder+'/delta_stage_15').to('cpu').detach()
# delta_s = delta_s.numpy()

img_delta = imshow(delta_s, imsize = 240)

delta_test = Image.open('./'+image_folder+'/stage13samplesize12_1.jpg').convert('L')
# image_delta = dest_img - src_img
# delta_test.show()
input("Press Enter to continue...")
src_img = Image.open('./'+image_folder+'/new_' + str(starting_stage) + '.jpg').convert('L') #
dest_img = ImageChops.add(img_delta, src_img)
# dest_img.show()
chrisimg = ImageChops.add(src_img, delta_test)
# chrisimg.show()

img1 = Image.open('./'+image_folder+'/new_13.jpg').convert('L')
img2 = Image.open('./'+image_folder+'/new_15.jpg').convert('L')
dimg1 = ImageChops.subtract_modulo(img2, img1)
dimg1.show()
testimg = ImageChops.add(img1, dimg1)
testimg.show()
