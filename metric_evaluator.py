from glob import glob 
import os
from solucion_omar.paper2 import read_etsp_file, filter_arcs_by_time_window, hybrid_sa_ts


pathsProblems = sorted(glob(os.path.join('Dataset', 'Problems', 'Problems', 'ETSPTW-MCR', 'G-E5-MCR(70%)', '*w120s*.txt')))
pathsSolution = sorted(glob(os.path.join('Dataset', 'Solutions', 'Solutions', 'ETSPTW-MCR', 'G-E5-MCR(70%)', '*w120s*.txt')))

def get_solution_cost(path):
    with open(path) as file:
        lines = file.readlines()
        return int(lines[1])
    

import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        problemPath = self.data[idx]
        optimalCost = get_solution_cost(self.labels[idx])

        data = read_etsp_file(problemPath)

        filtered_matrix, last_node = filter_arcs_by_time_window(data['nodes'], data['distance_matrix'])

        resp = hybrid_sa_ts(
        nodes=data['nodes'],
        distance_matrix=data['distance_matrix'],
        num_customers=data['num_customers'],
        battery_capacity=data['battery_capacity'],
        energy_rate=data['energy_rate'],
        max_iterations=15000,     
        T0=10000,                 
        alpha=0.95,               
        cooling_interval=100,     
        perturbation_interval=500, 
        R=30,                     
        tabu_tenure_moves=400,     
        tabu_tenure_stations=75)

        if resp == None:
            return np.nan, np.nan, np.nan, np.nan
        else:
    
            best_solution, best_cost, cost_progression, best_cost_progression  = resp

            gap = (best_cost - optimalCost)/optimalCost * 100

            return problemPath, optimalCost, best_cost, gap
        
    

dataset = MyDataset(pathsProblems, pathsSolution)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=25,
    pin_memory=False,
    drop_last=False
)

import numpy as np
from tqdm import tqdm

batchSize = dataloader.batch_size
results = np.full((len(dataset), 3), fill_value=np.nan, dtype=np.float32)

dataForDf = {
    'file_path':[],
    'optimal_cost':[], 
    'best_cost':[], 
    'gap':[]
}
i=0
for batch in tqdm(dataloader):
    file_path, optimal_cost, best_cost, gap =  batch
    print(batch)

    dataForDf['file_path'].append(file_path[0])
    dataForDf['optimal_cost'].append(optimal_cost.item())
    dataForDf['best_cost'].append(best_cost.item())
    dataForDf['gap'].append(gap.item())

    i+=1

import pandas as pd
results = pd.DataFrame(dataForDf)
print(results)

results.to_csv("results G-E5-MCR(70%) new algo.csv", index=False)