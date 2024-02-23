import torch
import torch_geometric
from torch_geometric.datasets import TUDataset,Planetoid
import torch.nn.functional as F
from tqdm import tqdm
import os
# from permutation import PermutationDataset
from utils import *
# from k_gnn import MyPreTransform
from torch_geometric.data import InMemoryDataset, Data


TUData=["PROTEINS","IMDB-BINARY","REDDIT-BINARY","COLLAB","NCI1","NCI109","DD"]
obgb=['ogbg-molhiv','ogbg-molpcba']

##zinc=[]


def file_initialize():
    if os.path.exists("./graph_level_dataset"):
        return    
    else:
        os.mkdir("./graph_level_dataset")
def load_dataset(dataset_name,model_name=None,permutation_type=None,index=None,dataset=None):
    file_initialize()
    file_name='./graph_level_dataset/'+dataset_name
    #print(file_name)
    if dataset_name in TUData:
        
        dataset = get_TUdataset(dataset_name,model_name=None,Permutation=permutation_type,dataset_anchored=dataset)
        print("Load Dataset: %s Successfully. "%(dataset_name)+(" Permutation Type :%s" %(permutation_type) if permutation_type is not None else ""))

    else:
        print("Error")
        raise Exception("Error in load_dataset")
    
    return dataset

def get_TUdataset(name, model_name=None, sparse=True, cleaned=False, normalize=False,Permutation=None,index=None, dataset_anchored=None):
    
    dataset = None
    if dataset_anchored is None:
        dataset = TUDataset(os.path.join('./graph_level_dataset', name), name, use_node_attr=True, cleaned=cleaned)
        dataset.data.edge_attr = None
    else:
        dataset = dataset_anchored.copy()

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    elif normalize:

        dataset.data.x -= torch.mean(dataset.data.x, axis=0)
        dataset.data.x /= torch.std(dataset.data.x, axis=0)

    if not sparse:
        max_num_nodes = 0
        for data in dataset:
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        if dataset.transform is None:
            dataset.transform = T.ToDense(max_num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(max_num_nodes)])
    
    return dataset

class TransformDataset(InMemoryDataset):
    """
    Address the data transformation for GNNML3 and KGNN
    """

    def __init__(self, cur_dataset, dataset_root, dataset_name, model_name, 
                 transform=None, pre_transform=None, pre_filter=None,permuted=None):

        self.cur_dataset = cur_dataset
        self.model_name=model_name
        if model_name== "GNNML3":
            root_name = dataset_root + dataset_name+"/GNNML3"
        elif model_name == "KGNN":
            root_name = dataset_root + dataset_name+"/K_GNN"
            #os.remove(root_name+"/")
        else:
            raise NotImplementedError("Error in TransformDataset constructor")
        
        if permuted is not None:
            root= os.path.join(root_name, str(permuted))
        else:
            root = os.path.join(root_name, "train")
        self.cur_dataset = cur_dataset
        self.name = dataset_name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.name=dataset_name

        self.num_i_2=0
        self.num_i_3=0

        if hasattr(cur_dataset, "get_idx_split"):
            self.get_idx_split = cur_dataset.get_idx_split()
        else:
            self.get_idx_split = None    

    def name(self):
        return self.name
    
    def get_idx_split(self):
        return self.get_idx_split

    @property
    def processed_file_names(self):
        return ['graph_data.pt']

    # def __len__(self) -> int:
    #     return len(self.label)
    def process(self):
        data_list = []
        if self.model_name == "GNNML3":
            for idx, data in enumerate(tqdm(self.cur_dataset)):
                data = SpectralDesign()(data)
                data_list.append(data)
                #print(type(data))
                
        elif self.model_name == "KGNN":
            for idx, data in enumerate(tqdm(self.cur_dataset)):
                data=MyPreTransform()(data)
                data_list.append(data)
        else:
            raise NotImplementedError("Check your dataset!")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data,slices), self.processed_paths[0])