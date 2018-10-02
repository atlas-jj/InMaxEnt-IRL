from PIL import Image, ImageDraw, ImageChops


image_folder = "robot_blocks/h1"

sample_size = 50
gap = 2


for i in range(sample_size-gap):
    img1 = Image.open('./'+image_folder+'/raw_' + str((i+1)) + '.jpg') #
    img2 = Image.open('./'+image_folder+'/raw_' + str((i+1+gap)) + '.jpg') #
    sub_img = ImageChops.subtract_modulo(img2, img1)
    sub_img.save('./'+image_folder+'/sub_'+str(int(i+1))+'.jpg')
    sub_img_inverse = ImageChops.subtract_modulo(img1, img2)
    sub_img_inverse.save('./'+image_folder+'/subinverse_'+str(int(i+1))+'.jpg')
