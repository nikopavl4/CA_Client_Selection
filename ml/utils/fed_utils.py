from torch.utils.data import random_split
from ml.fl.vehicle import Vehicle
from ml.fl.client import Client
import random

def create_fed_vehicles(trainset, nvehicles):
    """
    Splitting provided trainset into respective
    vehicles objects. Vehicles only carry datasets.
    They do not participate in Federated Process.
    """
    data_per_vehicle = len(trainset)/nvehicles
    split_list = [data_per_vehicle]*nvehicles
    split_list = [int(i) for i in split_list]
    vehicle_data = random_split(trainset, split_list)
    vehicle_list = []
    for i, dataset in enumerate(vehicle_data):
        new_vehicle = Vehicle(i, dataset)
        vehicle_list.append(new_vehicle)

    print(f'Successfully created {len(vehicle_list)} vehicles.')

    return vehicle_list

def update_fed_vehicles(client_list, new_dataset):
    """
    Adding new data samples to vehicles moving to another bs.
    Adding 50 new samples to every vehicle moving to another bs.
    """
    add_counter = 0
    if len(new_dataset) > 50:
        for client in client_list:
            for vehicle in client.vehicle_list:
                if (vehicle.current_bs != vehicle.previous_bs):
                    add_counter = add_counter+1
                    vehicle_newdata,  new_dataset= random_split(new_dataset, [50 , len(new_dataset)-50])
                    vehicle.add_samples(vehicle_newdata)
    else:
        print("No more available samples to add.")
    
    print(f'Successfully added samples to {add_counter} vehicles.')
    return client_list, new_dataset

def move_vehicles(vehicle_list, client_list, mobility):
    """
    Randomly moving a percentage of vehicles (defined by mobility)
    to other bs. Every bs should have at least 1 vehicle.
    """
    vehicles_to_move = random.sample(range(0,len(vehicle_list)), int(len(vehicle_list)*mobility))
    init_vehicles_to_move = vehicles_to_move.copy()
    
    for client in client_list:
        temp_vehicle_list = client.vehicle_list.copy()
        if len(client.vehicle_list) == 1:
            for vehicle in client.vehicle_list:
                client_list[vehicle.current_bs].reconfirm(vehicle)
            continue
        else:
            for vehicle in temp_vehicle_list:
                if vehicle.id in vehicles_to_move and len(client.vehicle_list)>1:
                    vehicles_to_move.remove(vehicle.id)
                    bs_to_go = random.randint(0,len(client_list)-1)
                    # if bs_to_go == vehicle.current_bs:
                    #     bs_to_go = bs_to_go + 1
                    
                    vehicle = client_list[vehicle.current_bs].unregister(vehicle)
                    client_list[bs_to_go].register(vehicle)
                else:
                    if vehicle.id not in init_vehicles_to_move:
                        client_list[vehicle.current_bs].reconfirm(vehicle)

    return client_list


def create_fed_clients(vehicle_list, nclients):
    """
    Randomly assigning vehicles to bs for the first time.
    Every bs has at least 1 vehicle.
    """    
    client_list = []

    for i in range(int(nclients)):
        new_client = Client(i)
        client_list.append(new_client)
    
    print(f'Successfully created {len(client_list)} clients.')
    for vehicle in vehicle_list:
        bs_to_assign = -1
        for client in client_list:
            if len(client.vehicle_list) == 0:
                bs_to_assign = client.id
                break
            else:
                continue
        if bs_to_assign == -1:
            bs_to_assign = random.randint(0,len(client_list)-1)
        
        client_list[bs_to_assign].register(vehicle)
    
    return client_list

def refresh_fed_clients(client_list):
    """
    Calling refresh function for all clients
    in order to recreate dataloaders with new samples.
    """
    for client in client_list:
        client.refresh()
    return client_list


def initialize_fed_clients(client_list, args, model):
    """
    Setting learning parameters to pass
    to the init_learning_parameter function
    of each client.
    """
    params = {'epochs':args.epochs, 'lr':args.lr, 'device':args.device, 'test_size':args.test_size, 'batch_size':args.batch_size, 'criterion':args.criterion, 'optimizer':args.optimizer}
    new_client_list = []
    for client in client_list:
        client.init_learning_parameters(params, model)
        new_client_list.append(client)

    print(f'Successfully initialized {len(new_client_list)} clients.')

    return new_client_list