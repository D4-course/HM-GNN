from networkx.classes.function import number_of_edges
from networkx.generators.random_graphs import barabasi_albert_graph
import numpy as np
from tqdm import tqdm
import argparse
import dgl
import dgl.data
import torch
from utils.gin import TwoGIN
from utils.ops import load_data
from sklearn.model_selection import KFold, StratifiedKFold
import random
import statistics as st
import pickle
from main import get_args, sep_data, load_subtensor
import streamlit as st

def show_prediction(outputs, y, i=0):
    _, preds = torch.max(outputs, dim=1)
    return preds[i].item(), y[i].item()

def train_and_evaluate(model, num_cliques, feat, labels, graphs, dataloader, edge_weight, device, index):
    with torch.no_grad():
        for _, (input_nodes, seeds, blocks) in enumerate(dataloader):
            selected_idx = seeds
            IDs = []
            for block in blocks:
                IDs.append(block.edata[dgl.EID])
            batch_graph = []
            for i in selected_idx:
                if i < len(graphs):
                    batch_graph.append(graphs[i])
            batch_inputs, batch_labels, batch_edge_weight = load_subtensor(feat, labels, edge_weight, IDs,
                                                            seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            batch_pred = model(blocks, batch_inputs, batch_edge_weight, batch_graph, num_cliques)
            pred, actual = show_prediction(batch_pred, batch_labels, index)
            break
    return pred, actual


st.title("Heterogeneous Motifs Granph Neural Network")

st.header("Select Calssification Task")
selected_data = st.selectbox(
    "Classification Task",
    ['MUTAG', 'PTC_MR', 'PROTEINS', 'NCI1'],
    index=3
)


CUDA_LAUNCH_BLOCKING=1
args = get_args()
device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
if selected_data == 'PROTEINS':
    number_of_graphs = 1113
    max_num_slider = 187
elif selected_data == 'PTC_MR':
    number_of_graphs = 344
    max_num_slider = 187
elif selected_data == 'MUTAG':
    number_of_graphs = 188
    max_num_slider = 168
elif selected_data == 'NCI1':
    number_of_graphs = 4110
    max_num_slider = 187


st.header("Select Index")
index = st.slider(
    "Select index of molecule to predict",
    min_value=0,
    max_value=max_num_slider,
    value=5,
    step=1
)
st.subheader("Data")

isPredict = st.button("Predict")


def main():
    with open('preprocessed_datasets/' + selected_data, 'rb') as input_file:
        g = pickle.load(input_file)
    num_cliques = int(g.number_of_nodes()) - number_of_graphs
    labels = g.ndata['labels']
    features = g.ndata['feat']
    in_feats = features.size()[1]

    edge_weight = g.edata['edge_weight'].to(device)

    g = g.to(device)
    node_features = features.to(device)
    labels.to(device)

    graphs, num_classes = load_data(selected_data, args.degree_as_tag)

    for step in range(1):
        train_idx, valid_idx = sep_data(labels[:number_of_graphs], 0)
        for i in range(1):
            train_index = train_idx[i]
            valid_index = valid_idx[i]
            train_mask = [True if x in train_index else False for x in range(int(g.num_nodes()))]
            train_mask = np.array(train_mask)
            valid_mask = [True if x in valid_index else False for x in range(int(g.num_nodes()))]
            valid_mask = np.array(valid_mask)
            g.ndata['train_mask'] = torch.from_numpy(train_mask).to(device)
            g.ndata['val_mask'] = torch.from_numpy(valid_mask).to(device)
            train_mask = g.ndata['train_mask'].to(device)
            valid_mask = g.ndata['val_mask'].to(device)

            train_nid = torch.nonzero(train_mask, as_tuple=True)[0].to(device)
            val_nid = torch.nonzero(valid_mask, as_tuple=True)[0].to(device)
            g = g.to(device)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                train_nid,
                sampler,
                device=device,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers
            )
            val_dataloader = dgl.dataloading.NodeDataLoader(
                g,
                val_nid,
                sampler,
                device=device,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers
            )

            gin = TwoGIN(args.l_num, 2, in_feats, graphs[0].node_features.shape[1], args.h_dim, 2, args.drop_n, args.drop_c, args.learn_eps, 'sum', 'sum').to(device)
            gin.load_state_dict(torch.load(f'saved_model/best_{selected_data}.pt'))


            index = np.random.randint(0, 10)
            pred, actual = train_and_evaluate(gin, num_cliques, node_features, labels, graphs, dataloader, edge_weight, device, index)

            print(f'Predicted Label: {pred}')
            print(f'Actual Label: {actual}')
            return pred, actual

if isPredict:
    pred, actual = main()
    st.write('Predicted Label:', pred)
    st.write('Ground Truth Label:', actual)