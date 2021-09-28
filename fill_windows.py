import glob
import cv2

# folder with semantic segmented images
sem_paths = sorted(glob.glob('/home/schober/carla/output_carla_town_03_2/out_sem/*.png'))

# output folder for the filled semantic segmented images
out_path = '/home/schober/carla/output_carla_town_03_2/sem_filled/'

# rgb color of the car class from carla
rgb_car = (0, 0, 142)

for sem_img_path in sem_paths:
    # load in the semantic segmented image
    sem_img = cv2.imread(sem_img_path)
    sem_img = cv2.cvtColor(sem_img, cv2.COLOR_BGR2RGB)

    # get binary mask of the class car
    mask = cv2.inRange(sem_img, rgb_car, rgb_car)
    # find contours in the binary image
    contour, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # fill out each contour
    for cnt in contour:
        cv2.drawContours(sem_img, [cnt], 0, rgb_car, -1)

    # save image in output folder
    file_name = sem_img_path.split('/')[-1]
    sem_img = cv2.cvtColor(sem_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path + file_name, sem_img)
