from PIL import Image, ImageDraw, ImageChops
import torch
from torchvision import datasets, models, transforms

def imshow(tensor, imsize=512, title=None):
    image = tensor.clone().cpu()
    image = image.view(*tensor.size())
    image = transforms.ToPILImage()(image).convert('L')
    return image

image_folder = "blocks/output4"
starting_index = 15
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
# image_delta = dest_img - src_img
img_delta.show()
input("Press Enter to continue...")
src_img = Image.open('./'+image_folder+'/new_' + str(starting_stage) + '.jpg').convert('L') #
dest_img = ImageChops.add_modulo(img_delta, src_img)
dest_img.show()
