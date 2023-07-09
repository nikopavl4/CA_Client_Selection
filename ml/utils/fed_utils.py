from torch.utils.data import random_split
from ml.fl.vehicle import Vehicle
import random

def create_fed_vehicles(trainset, nclients):
    """
    Splitting provided trainset into nclients respective
    vehicles objects.
    """
    data_per_vehicle = len(trainset)/nclients
    split_list = [data_per_vehicle]*nclients
    split_list = [int(i) for i in split_list]
    vehicle_data = random_split(trainset, split_list)
    vehicle_list = []
    for i, dataset in enumerate(vehicle_data):
        new_vehicle = Vehicle(i, dataset)
        vehicle_list.append(new_vehicle)

    print(f'Successfully created {len(vehicle_list)} vehicles.')

    return vehicle_list

def update_fed_vehicles(vehicle_list, new_dataset):
    """
    Adding new data samples to vehicles moving to another bs
    """
    add_counter = 0
    if len(new_dataset) > 50:
        for vehicle in  vehicle_list:
            if (vehicle.current_bs != vehicle.previous_bs) and vehicle.previous_bs != -1:
                add_counter = add_counter+1
                vehicle_newdata,  new_dataset= random_split(new_dataset, [50 , len(new_dataset)-50])
                vehicle.add_samples(vehicle_newdata)
    else:
        print("No more available samples to add.")
    print(f'Successfully added samples to {add_counter} vehicles.')
    return vehicle_list, new_dataset

def move_vehicles(vehicle_list, client_list):
    vehicles_to_move = random.sample(range(0,len(vehicle_list)), int(len(vehicle_list)*0.3))
    for vehicle in vehicle_list:
        if vehicle.id in vehicles_to_move:
            bs_to_go = random.randint(0,len(client_list))
            if bs_to_go == vehicle.current_bs:
                bs_to_go = bs_to_go + 1
            client_list[bs_to_go].register(vehicle)
            client_list[vehicle.current_bs].unregister(vehicle)

def create_fed_clients(vehicle_list, nclients):
    for 
                                      



def initialize_fed_clients(client_list, args, model):
    params = {'epochs':args.epochs, 'lr':args.lr, 'device':args.device, 'test_size':args.test_size, 'batch_size':args.batch_size, 'criterion':args.criterion, 'optimizer':args.optimizer}
    new_client_list = []
    for client in client_list:
        client.init_parameters(params, model)
        new_client_list.append(client)

    print(f'Successfully initialized {len(new_client_list)} clients.')

    return new_client_list