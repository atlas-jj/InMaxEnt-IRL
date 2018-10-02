from PIL import Image, ImageDraw
import os, fnmatch
# import matplotlib.pyplot as plt


def process_img(src, crop1, crop2, resize_w):
    cropped = src.crop((crop1[0], crop1[1], crop2[0], crop2[1]))
    #cropped.show()
    cropped = cropped.resize((resize_w, resize_w))
    #cropped.show()
    #input("Press Enter to continue...")
    return cropped

image_folder = "robot_blocks/h1"

pattern = "*.jpg"

# mask_top_left = [711,47]
# mask_bottom_right = [1493,829]

mask_top_left = [749,69]
mask_bottom_right = [1529,849]
# mask_top_left = [460,0]
# mask_bottom_right = [1390,930]

sample_size = 101
# listOfFiles = os.listdir('./'+image_folder)

img_names = []
# for entry in listOfFiles:
#     if fnmatch.fnmatch(entry, pattern):
#         sample_size += 1
#         img_names.append(entry)

for i in range(sample_size):
    img = Image.open('./'+image_folder+'/' + str(i+1) + '.jpg')#
    # idx = int(img_names[i].split('.')[0].split('_')[1])
    print('index:' + str(i))
    if ((i+1) % 2 ==0):
        cropped = process_img(img, mask_top_left, mask_bottom_right, 240)
        cropped.save('./'+image_folder+'/raw_'+str(int(i/2+1))+'.jpg')
