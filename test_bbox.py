import cv2

img_path = '/home/schober/carla/output_carla_town_03/darknet/data/obj/000180.jpg'
#bb_path =  '/home/schober/carla/output_carla_town_03/darknet/data/obj/000180.txt'
bb_path  = '/home/schober/carla/output_carla_town_03/labels_resized/000180.txt'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

with open(bb_path, 'r') as f:
    content  = f.readlines()
    for line in content:
        x_center = int(float(line.split()[1]) * 512)
        y_center = int(float(line.split()[2]) * 256)
        w = int(float(line.split()[3]) * 512)
        h = int(float(line.split()[4]) * 256)

        x_left = int(x_center - .5 * w)
        x_right = int(x_center + .5 * w)
        y_top = int(y_center - .5 * h)
        y_bot = int(y_center + .5 * h)

        img = cv2.rectangle(img,(x_left, y_top), (x_right, y_bot), color=(255,0,0), thickness = 2)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('bb_box_resize.png', img)
