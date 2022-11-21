# The Class of Generating Customized Bus Instances
class CustomizedBusDataset(Dataset):
    def __init__(self, num_samples, input_size, seed, all_station, travel_time_G, Loc_x, Loc_y, up_down_index, depot, home_id, work_id):
        super(CustomizedBusDataset, self).__init__()

        if max_load < max_demand:
            raise ValueError(':param max_load: must be > max_demand')
        if seed is None:
            seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.num_samples = num_samples
        # station location information and travel time information
        self.space_time_demand = torch.rand((num_samples, len(all_station), input_size))
        for i in range(num_samples):
            # station location information,x
            for j in range(len(space_time_demand[i][0])):
                space_time_demand[i][0][j] = Loc_x[j]
            # station location information,y
            for j in range(len(space_time_demand[i][1])):
                space_time_demand[i][1][j] = Loc_y[j]

            # according to travel time information, the information for each time window is generated as follows:
            for j in range(len(space_time_demand[i][2])):
                if 0 < j < up_down_index[0]:
                    arr_time = travel_time_G[home_id[j - 1]][work_id[int(problem_constraint1[i][0][j]) - 1]]
                    space_time_demand[i][2][j] = (travel_time_G[depot[0]][home_id[j - 1]] * random.randint(1, max(int((20 - arr_time) / travel_time_G[depot[0]][home_id[j - 1]] - 1), 1)) + random.randint(0, 5)) / self.max_time
                    problem_constraint1[i][1][j] = max(float(space_time_demand[i][2][j]) + (arr_time + random.randint(10, 15)) / self.max_time, 1)
                elif j == 0:
                    space_time_demand[i][2][j] = 0
                elif j >= up_down_index[0]:
                    space_time_demand[i][2][j] = 1

        # add passenger flow data according to the number of time window constraints
        self.od_shape = (num_samples, 1, input_size)
        self.loads = torch.full(dynamic_shape, 1.)


'''Call the files in folders "Sioux_0", "Sioux_1", and "Major" 
to generate training, validation and test data sets for the three road netwoks, respectively.'''
# Codes for customized bus route network instances, such as Sioux_0, Sioux_1, and Major.
all_stations = []  # store all stations in a load network
depot = []  # a CP
home_id = []  # boarding stations
work_id = [3, 4]  # alighting stations
transit_network = ''  # a customized bus route network
network_trip_time = ''  # travel time file
network_station = ''  # station location file

route_network = ''  # input a name for the load network

'''the load network: Sioux_0'''
if route_network == 'Sioux_0':
    transit_network = 'Sioux_0'
    network_trip_time = 'sioux_time'
    network_station = 'sioux_stops'
    # a CP
    depot = [19]
    # boarding stations
    home_id = [13, 14, 18, 20, 21, 22, 23]
    # alighting stations
    work_id = [3, 4]

'''the load network: Sioux_1'''
if route_network == 'Sioux_1':
    transit_network = 'Sioux_1'
    network_trip_time = 'sioux_time'
    network_station = 'sioux_stops'
    # a CP
    depot = [19]
    # boarding stations
    home_id = [9, 10, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23]
    # alighting stations
    work_id = [3, 4]

'''the load network: Major'''
if route_network == 'Major':
    transit_network = 'Major'
    network_trip_time = 'major_time'
    network_station = 'major_stops'
    # a CP
    depot = [105]
    # boarding stations
    home_id = [2, 3, 4, 11, 27, 28, 29, 30, 40, 68, 122, 124, 126, 154, 155, 156, 157, 198, 199, 200, 203, 204,
               225, 249, 250, 251, 252, 253, 254, 282, 283, 286]
    # alighting stations
    work_id = [24, 149, 151, 229]

# list 'up_station'  contains a CP and all boarding stations
up_station = copy.deepcopy(depot)
up_station.extend(copy.deepcopy(home_id))

# list 'all_station'  contains a CP, all boarding stations, and all alighting stations
all_station = copy.deepcopy(depot)
all_station.extend(copy.deepcopy(home_id))
all_station.extend(copy.deepcopy(work_id))

# obtain the location information of each station
travel_time_file = network_trip_time + '.csv'
stop_file = network_station + '.csv'
people_od = {}
Loc = []
Loc_x = []
Loc_y = []
down_Loc_x = []
down_Loc_y = []
with open(stop_file, 'r', encoding='utf-8') as f0:
    for row0 in csv.reader(f0):
        Loc.append([float(row0[1]), float(row0[2])])
Loc_x.append(Loc[depot[0]][0])
Loc_y.append(Loc[depot[0]][1])
for i in range(len(Loc)):
    if i in home_id:
        Loc_x.append(Loc[i][0])
        Loc_y.append(Loc[i][1])
for i in range(len(Loc)):
    if i in work_id:
        Loc_x.append(Loc[i][0])
        Loc_y.append(Loc[i][1])

# normalize the value of the two-dimensional coordinates of each station
max_loc_x = max(Loc_x)
max_loc_y = max(Loc_y)

Loc_x = [i / max(Loc_x) for i in Loc_x]
Loc_y = [i / max(Loc_y) for i in Loc_y]

# all_stations = depot + home_id + work_id

# Get the shortest travel time between any two stations
up_down_index = [len(depot) + len(home_id), len(work_id)]
travel_time_G = [[0 for i in range(len(Loc))] for i in range(len(Loc))]
with open(travel_time_file, 'r', encoding='utf-8') as f1:
    for row1 in csv.reader(f1):
        travel_time_G[int(row1[0])][int(row1[1])] = float(row1[2])

# Initially loaded environment information
parser.add_argument('--seed', default=12345, type=int)
parser.add_argument('--nodes', dest='num_nodes', default=up_down_index[0]+up_down_index[1], type=int)
parser.add_argument('--upnodes', dest='up_num_nodes', default=up_down_index[0], type=int)
parser.add_argument('--downnodes', dest='down_num_nodes', default=up_down_index[1], type=int)
parser.add_argument('--train-size',default=64000, type=int)
parser.add_argument('--valid-size', default=1280, type=int)
parser.add_argument('--testing-size', default=128, type=int)

args = parser.parse_args()
#  generating training instances
train_data = CustomizedBusDataset(args.train_size, args.num_nodes, args.seed, all_station, travel_time_G, Loc_x, Loc_y,
                                  up_down_index, depot, home_id, work_id)
#  generating validation instances
valid_data = CustomizedBusDataset(args.valid_size, args.num_nodes, args.seed + 1, all_station, travel_time_G,
                                  Loc_x, Loc_y, up_down_index, depot, home_id, work_id)
#  generating test instances
test_data = CustomizedBusDataset(args.testing_size, args.num_nodes, args.seed + 2, all_station, travel_time_G,
                                 Loc_x, Loc_y, up_down_index, depot, home_id, work_id)
