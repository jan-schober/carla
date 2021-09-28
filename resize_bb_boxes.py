import glob
import cv2

town = 'output_carla_town_03_1'

# root folder with images and bounding boxes
root_folder = '/home/schober/carla/output_carla_town_03_1/darknet/data/obj/'
# get list of bounding boxes and images
bb_boxes = sorted(glob.glob(root_folder + '*.txt'))
images = sorted(glob.glob(root_folder + '*.jpg'))

# folder with semantic images
sem_images = sorted(glob.glob('/home/schober/carla/output_carla_town_03_1/sem_filled/' + '*.png'))

# folder to store the scaled bounding boxes
out_folder = '/home/schober/carla/output_carla_town_03_1/labels_resized_new/'

# rgb color of semantic class car and pedestrian
rgb_car = (0, 0, 142)
rgb_pedestrian = (220, 20, 60)

# image width and height
image_width = 512
image_height = 256


def resize_box(x_c, y_c, w_px, h_px, image_mask):
    '''
    :param x_c: x-center of bounding box
    :param y_c: y-center of bounding box
    :param w_px: width of bounding box
    :param h_px: height of bounding box
    :param image_mask: binary image of class pedestrian or car
    :return: new bounding box coordinates
    '''
    x_left, x_right, y_top, y_bot = None, None, None, None
    # goes from left side of bounding box to x-center and gets back first hit with binary mask
    for x in range(int(x_c - .5 * w_px), x_c):
        if x <= 0:
            x_left = 0
            break
        elif x >= image_width:
            x_left = image_width
            break
        elif 255 in image_mask[:, x]:
            x_left = x
            break
        else:
            x_left = int(x_c - .5 * w_px)

    # goes from right side of bounding box to x-center and gets back first hit with binary mask
    for x in range(int(x_c + .5 * w_px), x_c, -1):
        if x <= 0:
            x_right = 0
            break
        elif x >= image_width:
            x_right = image_width
            break
        elif 255 in image_mask[:, x]:
            x_right = x
            break
        else:
            x_right = int(x_c + .5 * w_px)

    # goes from top side of bounding box to y-center and gets back first hit with binary mask
    for y in range(int(y_c - .5 * h_px), y_c, 1):
        if y <= 0:
            y_top = 0
            break
        elif y >= image_height:
            y_top = image_height
            break
        elif 255 in image_mask[y, :]:
            y_top = y
            break
        else:
            y_top = int(y_c - .5 * h_px)

    # goes from bottom side of bounding box to y-center and gets back first hit with binary mask
    for y in range(int(y_c + .5 * h_px), y_c, -1):
        if y <= 0:
            y_bot = 0
            break
        elif y >= image_height:
            y_bot = image_height
            break
        elif 255 in image_mask[y, :]:
            y_bot = y
            break
        else:
            y_bot = int(y_c + .5 * h_px)
    if x_left is None:
        x_left = int(x_c - .5 * w_px)
    if x_right is None:
        x_right = int(x_c + .5 * w_px)
    if y_top is None:
        y_top = int(y_c - .5 * h_px)
    if y_bot is None:
        y_bot = int(y_c + .5 * h_px)
    return x_left, x_right, y_top, y_bot


def main():
    for bb_path, img_path, sem_img_path in zip(bb_boxes, images, sem_images):
        # reads in the nomalized bounding box coordinates
        with open(bb_path) as f:
            content = f.readlines()

        # reads the semantic images
        sem_img = cv2.imread(sem_img_path)
        sem_img = cv2.cvtColor(sem_img, cv2.COLOR_BGR2RGB)
        # gets binary images from class car and pedestrian
        mask_car = cv2.inRange(sem_img, rgb_car, rgb_car)
        mask_pedestrian = cv2.inRange(sem_img, rgb_pedestrian, rgb_pedestrian)

        # gets file name for output file
        file_name = bb_path.split('/')[-1]
        file_name = file_name.split('.')[0]

        # converts the normalized bounding box coordinates to px coordinates and create an outfile txt document
        with open(out_folder + file_name + '.txt', 'a') as outfile:
            for line in content:
                cls = int(line.split()[0])
                x_center = int(float(line.split()[1]) * image_width)
                y_center = int(float(line.split()[2]) * image_height)
                w = int(float(line.split()[3]) * image_width)
                h = int(float(line.split()[4]) * image_height)
                if x_center <= 0:
                    x_center = 0
                elif x_center >= image_width:
                    x_center = image_width - 1
                if y_center <= 0:
                    y_center = 0
                elif y_center >= image_height:
                    y_center = image_height - 1

                # for class car and pedestrian the bounding box coordinates are rescaled
                if cls == 2:
                    x_l, x_r, y_t, y_b = resize_box(x_center, y_center, w, h, mask_car)
                elif cls == 4:
                    x_l, x_r, y_t, y_b = resize_box(x_center, y_center, w, h, mask_pedestrian)

                # convert the px coordinates back to normalized coordinates
                x_center_new = (x_l + (x_r - x_l) / 2) / image_width
                y_center_new = (y_t + (y_b - y_t) / 2) / image_height
                w_new = (x_r - x_l) / image_width
                h_new = (y_b - y_t) / image_height

                if x_center_new <= 0:
                    x_center = 0
                elif x_center >= 1:
                    x_center = 1
                if y_center <= 0:
                    y_center = 0
                elif y_center >= 1:
                    y_center = 1

                # write the new coordinates into outfile
                outfile.write(f"{cls} {x_center_new} {y_center_new} {w_new} {h_new} \n")

if __name__ == "__main__":
    main()
