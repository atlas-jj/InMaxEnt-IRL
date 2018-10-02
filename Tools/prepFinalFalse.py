from PIL import Image, ImageDraw, ImageChops
import os, fnmatch, sys
import torch
from torchvision import datasets, models, transforms

task_name = "plug"
dest = "finalfalse"
image_folder = "plug/"+dest
# mask_top_left = [711,47]
# mask_bottom_right = [1493,829]
reference_image = "./plug/p1/raw_49.jpg"
mask_top_left = [749,69]
mask_bottom_right = [1529,849]
# mask_top_left = [460,0]
# mask_bottom_right = [1390,930]



pattern = "*.jpg"
os.system("ffmpeg -i ./"+image_folder+"/output.avi ./"+image_folder+"/%d.jpg")
input_size = 240 # input raw image size, to the network

listOfFiles = os.listdir('./'+image_folder)
sample_size = 0
img_names = []
for entry in listOfFiles:
    if fnmatch.fnmatch(entry, pattern):
        sample_size += 1
        img_names.append(entry)

print("1. please define the final sample size you want to use:")
sample_size = int(input("Enter a number: "))

def process_img(src, crop1, crop2, resize_w):
    cropped = src.crop((crop1[0], crop1[1], crop2[0], crop2[1]))
    #cropped.show()
    cropped = cropped.resize((resize_w, resize_w))
    #cropped.show()
    #input("Press Enter to continue...")
    return cropped

print("2. please define raw sample gap, default 2:")
raw_gap = 2
raw_gap = int(input("Enter a number: "))
raw_sample_size = 0
for i in range(sample_size):
    img = Image.open('./'+image_folder+'/' + str(i+1) + '.jpg')#
    # idx = int(img_names[i].split('.')[0].split('_')[1])
    print('index:' + str(i))
    if ((i+1) % raw_gap ==0):
        raw_sample_size += 1
        cropped = process_img(img, mask_top_left, mask_bottom_right, 240)
        cropped.save('./'+image_folder+'/raw_'+str(int(i/raw_gap+1))+'.jpg')

# now delete all frame images
for i in range (len(img_names)):
    os.remove('./'+image_folder + '/' + img_names[i])
print("frame images deleted!")
print("3. please define a reference final image")


delta_sample_size = 0
for i in range(raw_sample_size):
    img1 = Image.open('./'+image_folder+'/raw_' + str((i+1)) + '.jpg') #
    img2 = Image.open(reference_image) #
    sub_img = ImageChops.subtract_modulo(img2, img1)
    sub_img.save('./'+image_folder+'/sub_'+str(int(i+1))+'.jpg')
    sub_img_inverse = ImageChops.subtract_modulo(img1, img2)
    sub_img_inverse.save('./'+image_folder+'/subinverse_'+str(int(i+1))+'.jpg')
    delta_sample_size += 1

print("delta sample generated!")
#  now save torch samples

samples = torch.zeros(delta_sample_size, 1, input_size, input_size)
samples_inverse = torch.zeros(delta_sample_size, 1, input_size, input_size)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
    ]
)

for i in range(delta_sample_size):
    img = Image.open('./'+image_folder+'/sub_' + str(i+1) + '.jpg').convert('L') #
    img_inverse = Image.open('./'+image_folder+'/subinverse_' + str(i+1) + '.jpg').convert('L') #
    # print('entry: ' + img_names[i] + ', index: ' + str(idx))
    samples[i] = transform(img)  #
    samples_inverse[i] = transform(img_inverse)

torch.save(samples, '/home/chris/fc/pyTorch_code/InMaxEnt_IRL_Pytorch/generate_dataset/'+ task_name+'_'+dest+'_delta_samples_' + str(raw_gap))
torch.save(samples_inverse, '/home/chris/fc/pyTorch_code/InMaxEnt_IRL_Pytorch/generate_dataset/'+task_name + '_'+ dest+ '_delta_samples_inverse_'+str(raw_gap))
print("torch sample saved!")
