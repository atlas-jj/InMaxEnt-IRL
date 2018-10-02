from PIL import Image, ImageDraw, ImageChops

image_folder = "blocks/output4"
stage = 13
# back -1, -2, -3 -4 -5 -6
# forward 1 2 3 4 5 6
sample_size = 12
img1 = Image.open('./'+image_folder+'/raw_' + str(stage) + '.jpg') #

save_count = 0
int_mid = int(sample_size/2)
for i in range(sample_size+1):
    if (i - int_mid) != 0:
        img2 = Image.open('./'+image_folder+'/raw_' + str(stage - int_mid + i) + '.jpg') #
        print(str(stage - int_mid + i) + '.jpg')
        sub_img = ImageChops.subtract_modulo(img2, img1)
        sub_img.save('./'+image_folder+'/stage'+str(stage)+'samplesize'+str(sample_size)+'_'+str(int(save_count+1))+'.jpg')
        save_count += 1
