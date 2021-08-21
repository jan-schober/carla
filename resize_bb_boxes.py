import glob
import cv2

root_folder = '/home/schober/carla/output_carla_town_03_2/darknet/data/obj/'
out_folder = '/home/schober/carla/output_carla_town_03_2/labels_resized/'
bb_boxes = sorted(glob.glob(root_folder + '*.txt'))
images = sorted(glob.glob(root_folder + '*.jpg'))
sem_images = sorted(glob.glob('/home/schober/carla/output_carla_town_03_2/sem_filled/' + '*.png'))

assert len(bb_boxes) == len(images), 'Different lenght of images and bb'

rgb_car = (0, 0, 142)
image_width = 512
image_height = 256


def resize_box(x_c, y_c, w_px, h_px, image_mask):
    x_left, x_right, y_top, y_bot = -1, -1, -1, -1

    for x in range(x_c, int(x_c - .5 * w_px), -1):
        if x <= 0 or x >= image_width:
            if x_left == -1:
                x_left = int(x_c - .5 * w_px)
                break
            else:
                break
        else:
            px_value = image_mask[y_c, x]
            if px_value == 255:
                x_left = x

    for x in range(x_c, int(x_c + .5 * w_px), 1):
        if x <= 0 or x >= image_width:
            if x_right == -1:
                x_right = int(x_c + .5 * w_px)
                break
            else:
                break
        else:
            px_value = image_mask[y_c, x]
            if px_value == 255:
                x_right = x

    for y in range(y_c, int(y_c - .5 * h_px), -1):
        if y <= 0 or y >= image_height:
            if y_top == -1:
                y_top = int(y_c - .5 * h_px)
                break
            else:
                break
        else:
            px_value = image_mask[y, x_c]
            if px_value == 255:
                y_top = y

    for y in range(y_c, int(y_c + .5 * h_px), 1):
        if y <= 0 or y >= image_height:
            if y_bot == -1:
                y_bot = int(y_c + .5 * h_px)
                break
            else:
                break
        else:
            px_value = image_mask[y, x_c]
            if px_value == 255:
                y_bot = y

    if x_left == -1:
        x_left = int(x_c - .5 * w_px)
    if x_right == -1:
        x_right = int(x_c + .5 * w_px)
    if y_top == -1:
        y_top = int(y_c - .5 * h_px)
    if y_bot == -1:
        y_bot = int(y_c + .5 * h_px)

    return x_left, x_right, y_top, y_bot


def main():
    for bb_path, img_path, sem_img_path in zip(bb_boxes, images, sem_images):
        with open(bb_path) as f:
            content = f.readlines()
        sem_img = cv2.imread(sem_img_path)
        sem_img = cv2.cvtColor(sem_img, cv2.COLOR_BGR2RGB)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.inRange(sem_img, rgb_car, rgb_car)

        file_name = bb_path.split('/')[-1]
        file_name = file_name.split('.')[0]

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
                x_l, x_r, y_t, y_b = resize_box(x_center, y_center, w, h, mask)

                img = cv2.rectangle(img, (x_l, y_t), (x_r, y_b), color=(0, 0, 255), thickness=1)
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

                outfile.write(f"{cls} {x_center_new} {y_center_new} {w_new} {h_new} \n")

        # out_image = out_folder + 'images/'+file_name+'.png'
        # cv2.imwrite(out_image, img)
        # sys.exit()


if __name__ == "__main__":
    main()
