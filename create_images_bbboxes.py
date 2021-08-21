import pathlib
import queue
import random
import carla_vehicle_annotator as cva
import carla

# Start server
# ./CarlaUE4.sh -carla-server
# ./CarlaUE4.sh -carla-server -quality-level=Epic
# srun --gres=gpu:1 /home/rigoll/carla/CarlaUE4.sh -carla-server -quality-level=Epic

num_images = 1000
inv_framerate = 5  # take image every x seconds
num_vehicles = 80
num_walkers = 150
image_size_h = 512
image_size_v = 256
fov = 60
#path_output = pathlib.Path("/home/schober/carla/output/bounding_box_cityscapes")
output_path = 'output_carla_town_03_2/'
server_hostname = "ESS-Shaxs"
server_port = 2000
fixed_delta_seconds = 0.1
world_name = "Town03"


def retrieve_data(sensor_queue, frame, timeout=1):
    while True:
        try:
            data = sensor_queue.get(True, timeout)
        except queue.Empty:
            return None
        if data.frame == frame:
            return data


try:
    client = carla.Client(server_hostname, server_port)
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    SetVehicleLightState = carla.command.SetVehicleLightState
    FutureActor = carla.command.FutureActor
    client.set_timeout(10.0)

    world = client.load_world(world_name)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)

    # add vehicles
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
    blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
    blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
    blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
    blueprints = [x for x in blueprints if not x.id.endswith('t2')]
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    num_vehicles = min(len(spawn_points), num_vehicles)
    print(f"Number vehicles: {num_vehicles}")

    batch = []
    for _ in range(num_vehicles):
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)
        if blueprint.has_attribute("driver_id"):
            driver_id = random.choice(blueprint.get_attribute("driver_id").recommended_values)
            blueprint.set_attribute("driver_id", driver_id)
        blueprint.set_attribute("role_name", "autopilot")

        spawn_point = spawn_points.pop(0)
        batch.append(SpawnActor(blueprint, spawn_point)
                     .then(SetAutopilot(FutureActor, True)))

    responses = client.apply_batch_sync(batch, True)
    vehicles_id_list = []
    for response in responses:
        vehicles_id_list.append(response.actor_id)

    ego_vehicle = world.get_actor(vehicles_id_list[0])

    # add walkers
    blueprints_walkers = world.get_blueprint_library().filter("walker.pedestrian.*")

    batch = []
    for _ in range(num_walkers):
        spawn_point = carla.Transform()
        spawn_point.location = world.get_random_location_from_navigation()
        blueprint = random.choice(blueprints_walkers)
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "false")
        batch.append(SpawnActor(blueprint, spawn_point))

    responses = client.apply_batch_sync(batch, True)
    walkers_id_list = []
    for response in responses:
        walkers_id_list.append(response.actor_id)

    walkers_list = world.get_actors(walkers_id_list)

    # walker controllers
    batch = []
    walker_controller_blueprint = world.get_blueprint_library().find("controller.ai.walker")
    for i in range(len(walkers_id_list)):
        batch.append(carla.command.SpawnActor(walker_controller_blueprint, carla.Transform(), walkers_id_list[i]))
    responses = client.apply_batch_sync(batch, True)
    walker_controllers_id_list = []
    for response in responses:
        walker_controllers_id_list.append(response.actor_id)

    walker_controllers_list = world.get_actors(walker_controllers_id_list)
    for walker_controller in walker_controllers_list:
        walker_controller.start()
        walker_controller.go_to_location(world.get_random_location_from_navigation())
        walker_controller.set_max_speed(1 + random.random())

    # rgb camera
    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{image_size_h}")
    cam_bp.set_attribute("image_size_y", f"{image_size_v}")
    cam_bp.set_attribute("fov", f"{fov}")
    cam_bp.set_attribute('sensor_tick', '0.1')
    cam_location = carla.Location(1.2, 0, 1.75)
    cam_rotation = carla.Rotation(0, 0, 0)
    cam_transform = carla.Transform(cam_location, cam_rotation)
    cam = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle,
                            attachment_type=carla.AttachmentType.Rigid)

    # semantic segmentation

    semsec_bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
    semsec_bp.set_attribute("image_size_x", f"{image_size_h}")
    semsec_bp.set_attribute("image_size_y", f"{image_size_v}")
    semsec_bp.set_attribute("fov", f"{fov}")
    semsec_bp.set_attribute('sensor_tick', '0.1')
    semsec_location = carla.Location(1.2, 0, 1.75)
    semsec_rotation = carla.Rotation(0, 0, 0)
    semsec_transform = carla.Transform(semsec_location, semsec_rotation)
    semsec = world.spawn_actor(semsec_bp, semsec_transform, attach_to=ego_vehicle,
                               attachment_type=carla.AttachmentType.Rigid)

    # Spawn LIDAR sensor
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    lidar_bp.set_attribute('sensor_tick', '0.1')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '1120000')
    lidar_bp.set_attribute('upper_fov', '60')
    lidar_bp.set_attribute('lower_fov', '-60')
    lidar_bp.set_attribute('range', '150')
    lidar_bp.set_attribute('rotation_frequency', '20')
    lidar_bp_location = carla.Location(1.2, 0, 1.75)
    lidar_bp_rotation = carla.Rotation(0, 0, 0)
    lidar_transform = carla.Transform(lidar_bp_location, lidar_bp_rotation)
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)

    tick_queue = queue.Queue()
    world.on_tick(tick_queue.put)
    tick_idx, cam_idx, lidar_idx, semsec_idx = 0, 1, 2, 3

    # initalize
    for _ in range(5):
        world.tick()

    cam_queue = queue.Queue()
    semsec_queue = queue.Queue()
    cam.listen(cam_queue.put)
    semsec.listen(semsec_queue.put)
    lidar_queue = queue.Queue()
    lidar.listen(lidar_queue.put)

    q_list = []
    q_list.append(tick_queue)
    q_list.append(cam_queue)
    q_list.append(lidar_queue)
    q_list.append(semsec_queue)

    frame = 0
    print("World Initalized")
    for i in range(num_images * int(inv_framerate / fixed_delta_seconds)):
        if i % int(inv_framerate / fixed_delta_seconds) == 0:
            nowFrame = world.tick()

            while cam_queue.qsize() != 1 or semsec_queue.qsize() != 1:
                pass

            data = [retrieve_data(q, nowFrame) for q in q_list]
            assert all(x.frame == nowFrame for x in data if x is not None)

            cam_image = data[cam_idx]
            semsec_image = data[semsec_idx]
            lidar_image = data[lidar_idx]
            snap = data[tick_idx]

            ## start of implementing the 2d bounding box generator

            # Attach additional information to the snapshot
            vehicles_raw = world.get_actors().filter('vehicle.*')
            walkers_raw = world.get_actors().filter('walker.pedestrian.*')

            #vehicles = cva.snap_processing(vehicles_raw, snap)
            walkers = cva.snap_processing(vehicles_raw, walkers_raw, snap)

            # Calculating visible bounding boxes
            filtered_out, _ = cva.auto_annotate_lidar(walkers, cam, lidar_image, show_img=cam_image,
                                                      json_path='class_json_file.txt')

            # Save the results
            cva.save_output(output_path, cam_image, semsec_image, filtered_out['bbox'], filtered_out['class'], save_patched=False,
                            out_format= 'No')

            save_darknet = True

            # Save the results to darknet format
            if save_darknet: cva.save2darknet(filtered_out['bbox'], filtered_out['class'], cam_image, output_path)


            '''
            # if cam_image and semsec_image:
            if i % int(inv_framerate / fixed_delta_seconds) == 0:
                cam_image.save_to_disk(str(path_output / f"{frame:07}_cam.png"))
                semsec_image.save_to_disk(str(path_output / f"{frame:07}_semsec.png"),
                                          carla.ColorConverter.CityScapesPalette)
                frame += 1
            '''

finally:
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_id_list])

    for walker_controller in walker_controllers_list:
        walker_controller.stop()
    client.apply_batch([carla.command.DestroyActor(x) for x in walkers_id_list])
    client.apply_batch([carla.command.DestroyActor(x) for x in walker_controllers_id_list])
    cam.stop()
    semsec.stop()
    lidar.stop()
