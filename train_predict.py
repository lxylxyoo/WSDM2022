import argparse
import traceback
import time
import copy
import os
import numpy as np
import dgl
import torch
import pandas as pd
from tgn import TGN
from data_preprocess import TemporalDatasetA, TemporalDatasetB, TemporalDataset
from dataloading import (FastTemporalEdgeCollator, FastTemporalSampler,
                         SimpleTemporalEdgeCollator, SimpleTemporalSampler,
                         TemporalEdgeDataLoader, TemporalSampler, TemporalEdgeCollator)

from sklearn.metrics import average_precision_score, roc_auc_score
from collections import Counter
from datetime import datetime
# set random Seed
np.random.seed(12345)
torch.manual_seed(12345)

def load_prior_prob():
    scale_prob_A = pd.read_csv("data/final_test_csvs/scale_prob_final_A.csv", header=None)[0].values.reshape([-1,1])
    scale_prob_B = pd.read_csv("data/final_test_csvs/scale_prob_final_B.csv", header=None)[0].values.reshape([-1,1])
    return scale_prob_A, scale_prob_B

def train_seed_generator(graph, args):
    """Generate Seed (Edges) based on some techniques
 
    [1] chronologically split, use fresh data (e.g. latest one year)
    [2] filter by edge type, here use the edge features dim=0, may use edge type explicitly
    """
 
    if args.edge_select == 'full':
        # use full edges without any filter strategies
        train_seed = torch.arange(graph.num_edges())
 
    elif args.edge_select == 'chronological':
        # use latest data for training
        ts_cut = datetime.timestamp(datetime.strptime(args.split_date, "%Y-%m-%d"))
        train_seed = torch.nonzero(
            graph.edata['timestamp'] >= ts_cut, 
            as_tuple=False
        ).squeeze()
 
    elif args.edge_select == 'etype_filter':
        # filter edges by edge type
 
        # here use edge features instand, or use graph.edata['etype'] explicitly
        edge_type_tensor = graph.edata['feats'][:, -3].long()
 
        # get target edge types:
        # [1] hot edge types topK (count)
        etype_cnt = Counter(edge_type_tensor.numpy())
        etype_sorted = sorted(etype_cnt.items(), key=lambda x: x[1], reverse=True)
        etype_selected = [x[0] for x in etype_sorted[:args.etype_topk]]
 
        # [2] hot edge types (percentage)， 用 pandas 处理会更方便一些
 
        tmp = edge_type_tensor.unsqueeze(-1).repeat(1, len(etype_selected))
        train_seed = (tmp == torch.tensor(etype_selected)).sum(-1).nonzero(as_tuple=False).squeeze()
 
    else:
        raise Exception(f"invalid edge select option: '{args.edge_select}', not implemented!")
 
    return train_seed

def sort_test_graph(g):
    timestamp = g.edata["timestamp"]
    feats = g.edata["feats"]
    label = g.edata["label"]
    edges = g.edges()
    nnode = g.num_nodes()

    timestamp, perm = timestamp.sort()
    feats = feats[perm]
    label = label[perm]
    edges = (edges[0][perm], edges[1][perm])

    g_new = dgl.graph(edges, num_nodes=nnode)
    g_new.edata['timestamp'] = timestamp
    g_new.edata['feats'] = feats
    g_new.edata['label'] = label
    print(g_new.edata)
    return g_new, perm

def augment_test(graph_new_node, g_s, g_e, n=10):
    feats = g_s.edata["feats"]
    label = g_s.edata["label"]
    times_start = g_s.edata["timestamp"]
    times_end = g_e.edata["timestamp"]
    edges = g_s.edges()
    nnode = g_s.num_nodes()
    g_list = []
    for i in range(1, n - 1):
        g_aug = dgl.graph(edges, num_nodes=nnode)
        g_aug.edata["feats"] = feats
        g_aug.edata["label"] = label
        g_aug.edata["timestamp"] = times_start + (times_end - times_start) // (n - 1) * i
        g_new = dgl.add_edges(graph_new_node, *g_aug.edges(), data=g_aug.edata)
        yield g_new
    

def train(model, dataloader, sampler, criterion, optimizer, batch_size, fast_mode):
    model.train()
    total_loss = 0
    batch_cnt = 0
    last_t = time.time()
    for _, positive_pair_g, negative_pair_g, blocks in dataloader:
        positive_pair_g = positive_pair_g.to(device)
        negative_pair_g = negative_pair_g.to(device)
        blocks = [block.to(device) for block in blocks]
        optimizer.zero_grad()
        pred_pos, pred_neg = model.embed(
            positive_pair_g, negative_pair_g, blocks)
        loss = criterion(pred_pos, torch.ones_like(pred_pos))
        loss += criterion(pred_neg, torch.zeros_like(pred_neg))
        total_loss += float(loss)*batch_size
        retain_graph = True if batch_cnt == 0 and not fast_mode else False
        loss.backward(retain_graph=retain_graph)
        optimizer.step()
        model.detach_memory()
        model.update_memory(positive_pair_g)
        if fast_mode:
            sampler.attach_last_update(model.memory.last_update_t)
        print("Batch: ", batch_cnt, "Time: ", time.time()-last_t, "Batch Loss", loss.item())
        last_t = time.time()
        batch_cnt += 1
        # print(model.embedding_attn.feat_emb_list[0].weight)
    return total_loss

def finetune(model, dataloader, sampler, criterion, optimizer, batch_size, fast_mode):
    model.train()
    total_loss = 0
    batch_cnt = 0
    last_t = time.time()
    for _, positive_pair_g, negative_pair_g, blocks in dataloader:
        positive_pair_g = positive_pair_g.to(device)
        negative_pair_g = negative_pair_g.to(device)
        blocks = [block.to(device) for block in blocks]
        optimizer.zero_grad()
        pred_pos, _ = model.embed(
            positive_pair_g, negative_pair_g, blocks)
        loss = criterion(pred_pos, positive_pair_g.edata['label'].reshape([-1,1]))
        total_loss += float(loss)*batch_size
        retain_graph = True if batch_cnt == 0 and not fast_mode else False
        loss.backward(retain_graph=retain_graph)
        optimizer.step()
        # model.detach_memory()
        # model.update_memory(positive_pair_g)
        # if fast_mode:
            # sampler.attach_last_update(model.memory.last_update_t)
        # print("Batch: ", batch_cnt, "Time: ", time.time()-last_t, "Batch Loss", loss.item())
        last_t = time.time()
        batch_cnt += 1
        # print(model.embedding_attn.feat_emb_list[0].weight)
    return total_loss


def test_val(model, dataloader, sampler, criterion, batch_size, fast_mode):
    model.eval()
    batch_size = batch_size
    total_loss = 0
    aps, aucs = [], []
    batch_cnt = 0
    y_pred_list = []
    with torch.no_grad():
        for i, (_, positive_pair_g, negative_pair_g, blocks) in enumerate(dataloader):
            positive_pair_g = positive_pair_g.to(device)
            negative_pair_g = negative_pair_g.to(device)
            blocks = [block.to(device) for block in blocks]
            pred_pos, _ = model.embed(
                positive_pair_g, negative_pair_g, blocks)
            y_pred = pred_pos.sigmoid().cpu()
            # model.update_memory(positive_pair_g)
            # if fast_mode:
            #     sampler.attach_last_update(model.memory.last_update_t)

            batch_cnt += 1
            y_pred_list.extend(y_pred.tolist())
    # return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())
    return np.array(y_pred_list)

def save_checkpoint(model, optimizer, sampler, epoch, dataset, ckpt_dir):
    checkpoint = {
                  'epoch':epoch, 
                  'model_state_dict': model.state_dict(), 
                  'optimizer_state_dict': optimizer.state_dict(),
                  'neighbors': sampler.neighbors,
                  'e_id': sampler.e_id,
                  'cur_e_id': sampler.cur_e_id,
                  '__assoc__': sampler.__assoc__,
                  'last_update': sampler.last_update
                 }
    torch.save(checkpoint, ckpt_dir + 'checkpoint_%s_%s.pt' % (dataset, str(epoch)))

def write_predict_result(filename, res):
    with open(filename, "w") as fout:
        fout.write("\n".join([str(i[0]) for i in res]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=50,
                        help='epochs for training on entire dataset')
    parser.add_argument("--batch_size", type=int,
                        default=200, help="Size of each batch")
    parser.add_argument("--embedding_dim", type=int, default=100,
                        help="Embedding dim for link prediction")
    parser.add_argument("--memory_dim", type=int, default=100,
                        help="dimension of memory")
    parser.add_argument("--temporal_dim", type=int, default=100,
                        help="Temporal dimension for time encoding")
    parser.add_argument("--memory_updater", type=str, default='gru',
                        help="Recurrent unit for memory update")
    parser.add_argument("--aggregator", type=str, default='last',
                        help="Aggregation method for memory update")
    parser.add_argument("--n_neighbors", type=int, default=10,
                        help="number of neighbors while doing embedding")
    parser.add_argument("--sampling_method", type=str, default='topk',
                        help="In embedding how node aggregate from its neighor")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of heads for multihead attention mechanism")
    parser.add_argument("--fast_mode", action="store_true", default=False,
                        help="Fast Mode uses batch temporal sampling, history within same batch cannot be obtained")
    parser.add_argument("--simple_mode", action="store_true", default=False,
                        help="Simple Mode directly delete the temporal edges from the original static graph")
    parser.add_argument("--num_negative_samples", type=int, default=1,
                        help="number of negative samplers per positive samples")
    parser.add_argument("--dataset", type=str, default="wikipedia",
                        help="dataset selection wikipedia/reddit")
    parser.add_argument("--k_hop", type=int, default=1,
                        help="sampling k-hop neighborhood")                        
    parser.add_argument("--predict", action="store_true", default=False,
                        help="predict mode")
    parser.add_argument("--load_from_ft", action="store_true", default=False,
                        help="Loading fine-tuned model, only useful for DatasetA")
    parser.add_argument("--final", action="store_true", default=False,
                        help="final mode")    
    parser.add_argument("--feat_emb_A", action="store_true", default=False,
                        help="embedding DatasetA feats")
    parser.add_argument("--feat_emb_dim", type=int, default=100,
                        help="DatasetA embedding feats dim")    
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="learning rate")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt",
                        help="ckpt path")
    parser.add_argument("--result_dir", type=str, default="result",
                        help="result path")
    parser.add_argument("--predict_model", type=str, default="checkpoint",
                        help="predict model name")
    parser.add_argument("--finetune_model", type=str, default="checkpoint",
                        help="finetune model name")
    parser.add_argument("--mode", type=str, default="pre-training",
                        help="pre-training/fine-tuning/predict")
    parser.add_argument("--edge_select", type=str, default="full",
                        help="")
    parser.add_argument("--etype_topk", type=int, default=20,
                        help="")
    parser.add_argument("--split_date", type=str, default="2016-05-01",
                        help="")                                    
    args = parser.parse_args()

    assert not (args.fast_mode and args.simple_mode), "you can only choose one sampling mode"
    if args.k_hop != 1:
        assert args.simple_mode, "this k-hop parameter only support simple mode"

    if args.dataset == 'DatasetA':
        data = TemporalDatasetA(mode="train_fill-1")
        graph_test_start = TemporalDatasetA(mode="input_start_fill-1" if args.predict else "final_start_fill-1" if args.final else "test_start_fill-1")
        graph_test_end = TemporalDatasetA(mode="input_end_fill-1" if args.predict else "final_end_fill-1" if args.final else "test_end_fill-1")
        if args.feat_emb_A:
            data.edata["feats"] =  data.edata["feats"].to(torch.long) + 1
            graph_test_start.edata["feats"] = graph_test_start.edata["feats"].to(torch.long) + 1
            graph_test_end.edata["feats"] = graph_test_end.edata["feats"].to(torch.long) + 1
    elif args.dataset == 'DatasetB':
        data = TemporalDatasetB(mode="train")
        print(data.edata)
        graph_test_start = TemporalDatasetB(mode="final_start")
        graph_test_end = TemporalDatasetB(mode="final_end")
    else:
        print("Warning Using Untested Dataset: "+args.dataset)
        data = TemporalDataset(args.dataset)
    
    scale_prob_A, scale_prob_B = load_prior_prob()
    ####################################
    '''
        修正测试集的timestamp未从小到大排序的问题
    '''
    graph_test_start, perm_test_start = sort_test_graph(graph_test_start)
    graph_test_end, perm_test_end = sort_test_graph(graph_test_end)
    perm_test_start, _ = zip(*sorted(enumerate(perm_test_start), key=lambda x: x[1]))
    perm_test_end, _ = zip(*sorted(enumerate(perm_test_end), key=lambda x: x[1]))
    perm_test_start, perm_test_end = list(perm_test_start), list(perm_test_end)

    ####################################

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_nodes = data.num_nodes()
    num_edges = data.num_edges()

    graph_new_node = data

    # Sampler Initialization
    ###########################
    data_plus_start = dgl.add_edges(graph_new_node, *graph_test_start.edges(), data=graph_test_start.edata)
    data_plus_end = dgl.add_edges(graph_new_node, *graph_test_end.edges(), data=graph_test_end.edata)
    ###########################
    if args.fast_mode:
        sampler = FastTemporalSampler(graph_new_node, k=args.n_neighbors)
        new_node_sampler_s = FastTemporalSampler(data_plus_start, k=args.n_neighbors)
        new_node_sampler_e = FastTemporalSampler(data_plus_end, k=args.n_neighbors)
        edge_collator = FastTemporalEdgeCollator
    else:
        raise Exception("Only Support Fast Mode")
    
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(
        k=args.num_negative_samples)

    train_seed = torch.arange(graph_new_node.num_edges())

    train_dataloader = TemporalEdgeDataLoader(
                                              graph_new_node,
                                              train_seed_generator(graph_new_node, args),
                                              sampler,
                                              batch_size=args.batch_size,
                                              negative_sampler=neg_sampler,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0,
                                              collator=edge_collator,
                                              g_sampling=None)

    
    test_seed_s = torch.arange(graph_new_node.num_edges(), data_plus_start.num_edges())

    test_start_dataloader = TemporalEdgeDataLoader(
                                                      data_plus_start,
                                                      test_seed_s,
                                                      new_node_sampler_s,
                                                      batch_size=args.batch_size,
                                                      negative_sampler=neg_sampler,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      num_workers=0,
                                                      collator=edge_collator,
                                                      g_sampling=None)
    test_seed_e = torch.arange(graph_new_node.num_edges(), data_plus_end.num_edges())
    test_end_dataloader = TemporalEdgeDataLoader(
                                                      data_plus_end,
                                                      test_seed_e,
                                                      new_node_sampler_e,
                                                      batch_size=args.batch_size,
                                                      negative_sampler=neg_sampler,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      num_workers=0,
                                                      collator=edge_collator,
                                                      g_sampling=None)
    if args.dataset == "DatasetA" and args.feat_emb_A:
        edge_dim = data.edata['feats'].shape[1] * args.feat_emb_dim
    else:
        edge_dim = data.edata['feats'].shape[1]

    num_node = data.num_nodes()

    model = TGN(edge_feat_dim=edge_dim,
                memory_dim=args.memory_dim,
                temporal_dim=args.temporal_dim,
                embedding_dim=args.embedding_dim,
                num_heads=args.num_heads,
                num_nodes=num_node,
                feat_emb_dim=args.feat_emb_dim if args.feat_emb_A else None,
                n_neighbors=args.n_neighbors,
                memory_updater_type=args.memory_updater,
                feat_emb_A=args.feat_emb_A)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Implement Logging mechanism
    f = open("logging.txt", 'w')
    if args.fast_mode:
        sampler.reset()

    ckpt_dir = "model/%s/%s/%s/" % (
                        args.dataset, 
                        "pre-training" if args.mode == "predict" and args.dataset == "DatasetB" else \
                        "fine-tuning"  if args.mode == "predict" and args.dataset == "DatasetA" and args.load_from_ft else \
                        "pre-training" if args.mode == "predict" and args.dataset == "DatasetA" and not args.load_from_ft else \
                        args.mode,
                        args.ckpt_dir
    )
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    result_dir = "result/%s/%s/%s/" % (args.dataset, args.mode, args.result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    try:
        if args.mode == "pre-training":
            for i in range(args.epochs):
                train_loss = train(model, train_dataloader, 
                                    sampler,
                                    criterion, optimizer, args.batch_size, args.fast_mode)
                save_checkpoint(model, optimizer, sampler, i, args.dataset, ckpt_dir)
                memory_checkpoint = model.store_memory()
                if args.fast_mode:
                    new_node_sampler_s.sync(sampler)
                    new_node_sampler_e.sync(sampler)
                    # #################
                    # 同步sampler的last_update_t
                    if args.dataset == "DatasetB":  # DatasetB效果好，DatasetA效果不好
                        new_node_sampler_s.attach_last_update(sampler.last_update)
                        new_node_sampler_e.attach_last_update(sampler.last_update)
                    # #################
                y_pred_s = test_val(
                    model, test_start_dataloader, new_node_sampler_s, criterion, args.batch_size, args.fast_mode)
                model.restore_memory(memory_checkpoint)
                y_pred_e = test_val(
                    model, test_end_dataloader, new_node_sampler_e, criterion, args.batch_size, args.fast_mode)
                y_true = graph_test_start.edata['label'].numpy()
                ################################################
                '''
                    将测试集的预测顺序恢复到之前
                '''
                y_pred_s = y_pred_s[perm_test_start]
                y_pred_e = y_pred_e[perm_test_end]
                y_true = y_true[perm_test_start]
                #################################################
                if not args.predict:
                    ap1, auc1 = average_precision_score(y_true, y_pred_s), roc_auc_score(y_true, y_pred_s)
                    ap2, auc2 = average_precision_score(y_true, y_pred_e), roc_auc_score(y_true, y_pred_e)
                    ap3, auc3 = average_precision_score(y_true, y_pred_e - y_pred_s), roc_auc_score(y_true, y_pred_e - y_pred_s)
                    ap4, auc4 = average_precision_score(y_true, (y_pred_e + y_pred_s) / 2), roc_auc_score(y_true, (y_pred_e + y_pred_s) / 2)
                else:
                    if args.dataset == "DatasetB":
                        write_predict_result(result_dir + "output_%s_final_epoch_%s.csv" % (args.dataset[-1], str(i)), scale_prob_B * (y_pred_e + y_pred_s) / 2)
                    else:
                        write_predict_result(result_dir + "output_%s_middle_epoch_%s.csv" % (args.dataset[-1], str(i)), (y_pred_e + y_pred_s) / 2)
                    
                log_content = []
                log_content.append("Epoch: {}; Training Loss: {}\n".format(
                    i, train_loss))
                # log_content.append(
                    # "Epoch: {}; Test AP: {:.3f} AUC: {:.3f}\n".format(i, test_ap, test_auc))

                f.writelines(log_content)
                model.reset_memory(device)
                if i < args.epochs-1 and args.fast_mode:
                    sampler.reset()

                print(log_content[0])
                print("========Training is Done========")
        elif args.mode == "fine-tuning":
            if args.dataset == "DatasetA":
                load_dir = "model/%s/%s/%s/" % (args.dataset, "pre-training", args.ckpt_dir)
                if not os.path.exists(load_dir):
                    raise Exception(f"load_dir {load_dir} not exists")
                finetune_model_dir = os.path.join(load_dir, args.finetune_model)
                print("Loading pre-training model from :", finetune_model_dir)
                state_dict = torch.load(finetune_model_dir, map_location=torch.device("cpu"))
                model.load_state_dict({key:value.to(device) for key,value in state_dict['model_state_dict'].items()})
                graph_initial_start = TemporalDatasetA("test_start_fill-1")
                graph_initial_end = TemporalDatasetA("test_end_fill-1")
                graph_initial = copy.deepcopy(graph_initial_start)
                # 取start和end的中点
                graph_initial.edata["timestamp"] = torch.ceil((graph_initial_start.edata["timestamp"] + graph_initial_end.edata["timestamp"]) / 2)
                graph_initial_finetune = dgl.add_edges(graph_new_node, *graph_initial.edges(), data=graph_initial.edata)
                train_finetune_seed = torch.arange(graph_new_node.num_edges(), graph_initial_finetune.num_edges())
                sampler_ft = FastTemporalSampler(graph_initial_finetune, k=args.n_neighbors)
                sampler_ft.neighbors = state_dict['neighbors']
                sampler_ft.e_id = state_dict['e_id']
                sampler_ft.__assoc__ = state_dict['__assoc__']
                sampler_ft.cur_e_id = state_dict['cur_e_id']
                sampler_ft.last_update = state_dict['last_update']
                train_finetune_dataloader = TemporalEdgeDataLoader(
                                                          graph_initial_finetune,
                                                          train_finetune_seed,  # edges
                                                          sampler_ft,
                                                          batch_size=args.batch_size,
                                                          negative_sampler=neg_sampler,
                                                          shuffle=False,
                                                          drop_last=False,
                                                          num_workers=0,
                                                          collator=edge_collator,
                                                          g_sampling=None)

                finetune_plus_start = dgl.add_edges(graph_initial_finetune,  *graph_test_start.edges(), data=graph_test_start.edata)
                finetune_plus_end = dgl.add_edges(graph_initial_finetune,  *graph_test_end.edges(), data=graph_test_end.edata)
                new_node_sampler_s = FastTemporalSampler(finetune_plus_start, k=args.n_neighbors)
                new_node_sampler_e = FastTemporalSampler(finetune_plus_end, k=args.n_neighbors)                
                test_seed_s = torch.arange(graph_initial_finetune.num_edges(), finetune_plus_start.num_edges())
                test_start_dataloader = TemporalEdgeDataLoader(
                                                                  # data_plus_start,
                                                                  finetune_plus_start,
                                                                  test_seed_s,
                                                                  new_node_sampler_s,
                                                                  batch_size=args.batch_size,
                                                                  negative_sampler=neg_sampler,
                                                                  shuffle=False,
                                                                  drop_last=False,
                                                                  num_workers=0,
                                                                  collator=edge_collator,
                                                                  g_sampling=None)
                test_seed_e = torch.arange(graph_initial_finetune.num_edges(), finetune_plus_end.num_edges())
                test_end_dataloader = TemporalEdgeDataLoader(
                                                                  # data_plus_end,
                                                                  finetune_plus_end,
                                                                  test_seed_e,
                                                                  new_node_sampler_e,
                                                                  batch_size=args.batch_size,
                                                                  negative_sampler=neg_sampler,
                                                                  shuffle=False,
                                                                  drop_last=False,
                                                                  num_workers=0,
                                                                  collator=edge_collator,
                                                                  g_sampling=None)
                
                #############################################################################
                '''
                    直接在fine-tuning过程中输出final test的结果
                '''
                graph_final_start = TemporalDatasetA("final_start_fill-1")
                graph_final_end = TemporalDatasetA("final_end_fill-1")
                ######################
                '''
                    修正测试集的timestamp未从小到大排序的问题
                '''
                graph_final_start, perm_final_start = sort_test_graph(graph_final_start)
                graph_final_end, perm_final_end = sort_test_graph(graph_final_end)
                ######################
                finetune_plus_final_start = dgl.add_edges(graph_initial_finetune, *graph_final_start.edges(), data=graph_final_start.edata)
                finetune_plus_final_end = dgl.add_edges(graph_initial_finetune, *graph_final_end.edges(), data=graph_final_end.edata)
                new_node_sampler_final_s = FastTemporalSampler(finetune_plus_final_start, k=args.n_neighbors)
                new_node_sampler_final_e = FastTemporalSampler(finetune_plus_final_end, k=args.n_neighbors) 
                test_seed_final_s = torch.arange(graph_initial_finetune.num_edges(), finetune_plus_final_start.num_edges())
                test_final_start_dataloader = TemporalEdgeDataLoader(
                                                                  # data_plus_start,
                                                                  finetune_plus_final_start,
                                                                  test_seed_final_s,
                                                                  new_node_sampler_final_s,
                                                                  batch_size=args.batch_size,
                                                                  negative_sampler=neg_sampler,
                                                                  shuffle=False,
                                                                  drop_last=False,
                                                                  num_workers=0,
                                                                  collator=edge_collator,
                                                                  g_sampling=None)
                test_seed_final_e = torch.arange(graph_initial_finetune.num_edges(), finetune_plus_final_end.num_edges())
                test_final_end_dataloader = TemporalEdgeDataLoader(
                                                                  # data_plus_start,
                                                                  finetune_plus_final_end,
                                                                  test_seed_final_e,
                                                                  new_node_sampler_final_e,
                                                                  batch_size=args.batch_size,
                                                                  negative_sampler=neg_sampler,
                                                                  shuffle=False,
                                                                  drop_last=False,
                                                                  num_workers=0,
                                                                  collator=edge_collator,
                                                                  g_sampling=None)
                ############################################################################
                
                criterion_ft = torch.nn.BCEWithLogitsLoss()
                optimizer_ft = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
                for i in range(args.epochs):
                    finetune_loss = finetune(model, train_finetune_dataloader, 
                                        sampler_ft,
                                        criterion_ft, optimizer_ft, args.batch_size, args.fast_mode)
                    # save_checkpoint(model, optimizer_ft, sampler_ft, i, args.dataset, ckpt_dir)
                    memory_checkpoint = model.store_memory()
                    if args.fast_mode:
                        new_node_sampler_s.sync(sampler_ft)
                        new_node_sampler_e.sync(sampler_ft)
                        new_node_sampler_final_s.sync(sampler_ft)
                        new_node_sampler_final_e.sync(sampler_ft)
                        ##################
                        # 同步sampler的last_update_t
                        # new_node_sampler_s.attach_last_update(sampler_ft.last_update)
                        # new_node_sampler_e.attach_last_update(sampler_ft.last_update)            
                        # new_node_sampler_final_s.attach_last_update(sampler_ft.last_update)
                        # new_node_sampler_final_e.attach_last_update(sampler_ft.last_update)
                        ##################

#                     print("**** start y_pred_s ****")
#                     y_pred_s = test_val(
#                         model, test_start_dataloader, new_node_sampler_s, criterion_ft, args.batch_size, args.fast_mode)
#                     model.restore_memory(memory_checkpoint)
#                     print("**** start y_pred_e ****")
#                     y_pred_e = test_val(
#                         model, test_end_dataloader, new_node_sampler_e, criterion_ft, args.batch_size, args.fast_mode)
#                     y_true = graph_test_start.edata['label'].numpy()
#                     ################################################
#                     '''
#                         将测试集的预测顺序恢复到之前
#                     '''
#                     y_pred_s = y_pred_s[perm_test_start]
#                     y_pred_e = y_pred_e[perm_test_end]
#                     y_true = y_true[perm_test_start]
#                     #################################################
#                     if not args.predict:
#                         ap1, auc1 = average_precision_score(y_true, y_pred_s), roc_auc_score(y_true, y_pred_s)
#                         ap2, auc2 = average_precision_score(y_true, y_pred_e), roc_auc_score(y_true, y_pred_e)
#                         ap3, auc3 = average_precision_score(y_true, y_pred_e - y_pred_s), roc_auc_score(y_true, y_pred_e - y_pred_s)
#                         ap4, auc4 = average_precision_score(y_true, (y_pred_e + y_pred_s) / 2), roc_auc_score(y_true, (y_pred_e + y_pred_s) / 2)
#                         print(ap1, auc1)
#                         print(ap2, auc2)
#                         print(ap3, auc3)
#                         print(ap4, auc4)
#                     else:                    
#                         write_predict_result(result_dir + "output_%s_input_epoch_%s.csv" % (args.dataset[-1], str(i)), (y_pred_e + y_pred_s) / 2)



                    #########################################################
                    #########################################################
                    '''
                        First Middle Then Final
                    '''
                    print("**** start y_pred_final_s ****")
                    model.restore_memory(memory_checkpoint)
                    y_pred_final_s = test_val(
                        model, test_final_start_dataloader, new_node_sampler_final_s, criterion_ft, args.batch_size, args.fast_mode)
                    model.restore_memory(memory_checkpoint)
                    print("**** start y_pred_final_e ****")
                    y_pred_final_e = test_val(
                        model, test_final_end_dataloader, new_node_sampler_final_e, criterion_ft, args.batch_size, args.fast_mode)
                    ##################
                    '''
                        将最终测试集的预测顺序恢复到之前
                    '''
                    y_pred_final_s = y_pred_final_s[perm_final_start]
                    y_pred_final_e = y_pred_final_e[perm_final_end]
                    ##################
                    write_predict_result(result_dir + "output_%s_final_epoch_%s.csv" % (args.dataset[-1], str(i)), scale_prob_A * (y_pred_final_e + y_pred_final_s) / 2) 
                    #########################################################
                    #########################################################


                    log_content = []
                    log_content.append("Epoch: {}; Fine-tuning Loss: {}\n".format(
                        i, finetune_loss))
                    f.writelines(log_content)
                    # model.reset_memory(device)
                    if i < args.epochs-1 and args.fast_mode:
                        sampler_ft.reset()
                    print(log_content[0])
            else:
                print("Don't fine-tune for DatasetB, bad performance.")
                exit(1)
            print("========Fine-tunining {} is Done========".format(args.dataset))
        elif args.mode == "predict":
            predict_model_dir = os.path.join(ckpt_dir, args.predict_model)
            print("Loading predict model from :", predict_model_dir)
            state_dict = torch.load(predict_model_dir, map_location=torch.device("cpu"))
            model.load_state_dict({key:value.to(device) for key,value in state_dict['model_state_dict'].items()})
            sampler.neighbors = state_dict['neighbors']
            sampler.e_id = state_dict['e_id']
            sampler.__assoc__ = state_dict['__assoc__']
            sampler.cur_e_id = state_dict['cur_e_id']
            sampler.last_update = state_dict['last_update']
            ################################################################################################################
            if args.dataset == "DatasetA" and args.load_from_ft:
                graph_initial_start = TemporalDatasetA("test_start_fill-1")
                graph_initial_end = TemporalDatasetA("test_end_fill-1")

                graph_initial = copy.deepcopy(graph_initial_start)
                # 取start和end的中点
                graph_initial.edata["timestamp"] = torch.ceil((graph_initial_start.edata["timestamp"] + graph_initial_end.edata["timestamp"]) / 2)
                graph_initial_finetune = dgl.add_edges(graph_new_node, *graph_initial.edges(), data=graph_initial.edata)
                finetune_plus_start = dgl.add_edges(graph_initial_finetune,  *graph_test_start.edges(), data=graph_test_start.edata)
                finetune_plus_end = dgl.add_edges(graph_initial_finetune,  *graph_test_end.edges(), data=graph_test_end.edata)
                new_node_sampler_s = FastTemporalSampler(finetune_plus_start, k=args.n_neighbors)
                new_node_sampler_e = FastTemporalSampler(finetune_plus_end, k=args.n_neighbors)                
                test_seed_s = torch.arange(graph_initial_finetune.num_edges(), finetune_plus_start.num_edges())
                test_start_dataloader = TemporalEdgeDataLoader(
                                                                    # data_plus_start,
                                                                    finetune_plus_start,
                                                                    test_seed_s,
                                                                    new_node_sampler_s,
                                                                    batch_size=args.batch_size,
                                                                    negative_sampler=neg_sampler,
                                                                    shuffle=False,
                                                                    drop_last=False,
                                                                    num_workers=0,
                                                                    collator=edge_collator,
                                                                    g_sampling=None)
                test_seed_e = torch.arange(graph_initial_finetune.num_edges(), finetune_plus_end.num_edges())
                test_end_dataloader = TemporalEdgeDataLoader(
                                                                    # data_plus_end,
                                                                    finetune_plus_end,
                                                                    test_seed_e,
                                                                    new_node_sampler_e,
                                                                    batch_size=args.batch_size,
                                                                    negative_sampler=neg_sampler,
                                                                    shuffle=False,
                                                                    drop_last=False,
                                                                    num_workers=0,
                                                                    collator=edge_collator,
                                                                    g_sampling=None)
            ################################################################################################################
            if args.fast_mode:
                new_node_sampler_s.sync(sampler)
                new_node_sampler_e.sync(sampler)
                ##################
                # 同步sampler的last_update_t
                if args.dataset == "DatasetB":
                    new_node_sampler_s.attach_last_update(sampler.last_update)
                    new_node_sampler_e.attach_last_update(sampler.last_update)
                ##################
            memory_checkpoint = model.store_memory()
            y_pred_s = test_val(
                model, test_start_dataloader, new_node_sampler_s, criterion, args.batch_size, args.fast_mode)
            model.restore_memory(memory_checkpoint)
            y_pred_e = test_val(
                model, test_end_dataloader, new_node_sampler_e, criterion, args.batch_size, args.fast_mode)
            ##################
            '''
                将中期测试集/最终测试集的预测顺序恢复到之前
            '''
            y_pred_s = y_pred_s[perm_test_start]
            y_pred_e = y_pred_e[perm_test_end]
            ##################
            '''
            ################################################################################################################
            '''
                #start和end之间取均匀 n 个timestamp进行预测，然后求n个结果（包含start和end)的最大/平均
                #结论：效果基本不变
            '''
            del data_plus_start
            del data_plus_end
            y_pred_x_list = []
            for i, data_plus_x in enumerate(augment_test(graph_new_node, graph_test_start, graph_test_end, n=10)):
                print("Predicting x=%d" % i)
                new_node_sampler_x = FastTemporalSampler(data_plus_x, k=args.n_neighbors)
                test_seed_x = torch.arange(graph_new_node.num_edges(), data_plus_x.num_edges())
                test_x_dataloader = TemporalEdgeDataLoader(
                                                                data_plus_x,
                                                                test_seed_x,
                                                                new_node_sampler_x,
                                                                batch_size=args.batch_size,
                                                                negative_sampler=neg_sampler,
                                                                shuffle=False,
                                                                drop_last=False,
                                                                num_workers=0,
                                                                collator=edge_collator,
                                                                g_sampling=None)
                model.restore_memory(memory_checkpoint)
                if args.fast_mode:
                    new_node_sampler_x.sync(sampler)
                    if args.dataset == "DatasetB":
                        new_node_sampler_x.attach_last_update(sampler.last_update)
                y_pred_x = test_val(
                    model, test_x_dataloader, new_node_sampler_x, criterion, args.batch_size, args.fast_mode)     
                y_pred_x_list.append(y_pred_x.reshape(-1,1))                                                       
            if args.predict:
                write_dir = result_dir + "output_%s_input_mean_epoch_%s.csv" % (args.dataset[-1], state_dict['epoch'])    
            elif args.final:
                write_dir = result_dir + "output_%s_final_mean_epoch_%s.csv" % (args.dataset[-1], state_dict['epoch'])
            max_pred = np.concatenate([y_pred_s.reshape([-1,1]), *y_pred_x_list, y_pred_e.reshape([-1,1])], axis=1).mean(axis=1, keepdims=True)
            write_predict_result(write_dir, max_pred)
            print("Writing predict result to {} is done".format(write_dir))
            ################################################################################################################
            '''
            if args.predict:
                write_dir = result_dir + "output_%s_input_e+s_epoch_%s_fixsort.csv" % (args.dataset[-1], state_dict['epoch'])
            elif args.final:
                write_dir = result_dir + "output_%s_final_e+s_epoch_%s.csv" % (args.dataset[-1], state_dict['epoch'])
            
            #####
            write_predict_result(write_dir, (y_pred_e + y_pred_s) / 2)
            ######
            # write_predict_result(write_dir, (y_pred_e + y_pred_s) / 2)
            print("Writing predict result to {} is done".format(write_dir))
    except:
        traceback.print_exc()
        error_content = "Training Interreputed!"
        f.writelines(error_content)
        f.close()
        # exit(-1)
