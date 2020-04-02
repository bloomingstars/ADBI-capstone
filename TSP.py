from collections import namedtuple
import numpy as np
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def generate_graph_matrix(n=10, size=1):
    coords = size * np.random.uniform(size=(n,2))

coords = 1 * np.random.uniform(size=(10,2))

State= namedtuple('State',['cities','coords','visited cities'])

def tensor_creation(state):
    # tensor with the foll info:
    #presence in visited_cities
    #whether beg node
    #whether end node
    #coord_x
    #coord_y
    solution =set(state.visited_cities)
    num_states=coords.shape[0]
    tensor_for_all_states=[[
        (1 if this_state in solution else 0),
        (0 if solution.length<=0 or this_state!=solution[-1] else 1),
        (0 if solution.length <= 0 or this_state != solution[0] else 1),
        (coords[this_state,0]),
        (coords[this_state,1])
    ]for this_state in range(num_states)]

    return torch.tensor(tensor_for_all_states,dtype=torch.float32, requires_grad=False, device=device)

torch_creation()