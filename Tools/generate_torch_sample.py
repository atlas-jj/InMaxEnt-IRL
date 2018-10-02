from PIL import Image, ImageDraw
import torch
from torchvision import datasets, models, transforms

dest = "h1"
image_folder = "robot_blocks/"+dest

sample_size = 48
input_size = 240
samples = torch.zeros(sample_size, 1, input_size, input_size)
samples_inverse = torch.zeros(sample_size, 1, input_size, input_size)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
    ]
)

for i in range(sample_size):
    img = Image.open('./'+image_folder+'/sub_' + str(i+1) + '.jpg').convert('L') #
    img_inverse = Image.open('./'+image_folder+'/subinverse_' + str(i+1) + '.jpg').convert('L') #
    # print('entry: ' + img_names[i] + ', index: ' + str(idx))
    samples[i] = transform(img)  #
    samples_inverse[i] = transform(img_inverse)

torch.save(samples, './'+image_folder+'/'+dest+'_delta_samples_2')
torch.save(samples_inverse, './'+image_folder+'/'+ dest+ '_delta_samples_inverse_2')
