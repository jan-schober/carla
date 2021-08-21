import glob
import cv2

sem_paths = sorted(glob.glob('/home/schober/carla/output_carla_town_03_2/out_sem/*.png'))

out_path = '/home/schober/carla/output_carla_town_03_2/sem_filled/'
rgb_car = (0, 0, 142)

for sem_img_path in sem_paths:

    sem_img = cv2.imread(sem_img_path)
    sem_img = cv2.cvtColor(sem_img, cv2.COLOR_BGR2RGB)

    mask = cv2.inRange(sem_img, rgb_car, rgb_car)

    contour, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(sem_img, [cnt], 0, rgb_car, -1)

    file_name = sem_img_path.split('/')[-1]
    sem_img = cv2.cvtColor(sem_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path+file_name,sem_img)
