import os
import warnings
import pickle
import datetime
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import time
from transformers import AutoTokenizer, AutoModel

from utils import *

warnings.filterwarnings('ignore')


def stats(data):
    author_list = [i for i in data["node_feat"]]
    author_set = set(author_list)
    num_authors = len(data["node_feat"])
    num_edges = len(data["graph"])
    avg_degree = (num_edges * 2) / num_authors

    author_2024 = 0
    paper_2024 = [0, 0, 0, 0, 0]
    paper_total = 0
    for author, author_papers in data["node_feat"].items():
        author_latest = False
        for author_paper in author_papers:
            pub = author_paper["published"].split("-")
            pub_year = int(pub[0])
            pub_month = int(pub[1])
            paper_total += 1
            if pub_year == 2024:
                author_latest = True
                paper_2024[pub_month - 1] += 1

        if author_latest:
            author_2024 += 1

    print("Num of Authors: {}, Num of Edges: {}, Avg. Degree: {}, Author 2024: {}, Paper Total: {}, Period Slot: "
          "{}/{}/{}/{}/{}:{}".format(num_authors, num_edges, avg_degree, author_2024, paper_total, paper_2024[0],
                                     paper_2024[1], paper_2024[2], paper_2024[3], paper_2024[4], sum(paper_2024)))

    return paper_total


def reindex(data):
    author2id = dict()
    id2author = dict()
    author_list = [i for i in data["node_feat"]]
    for aid, author in enumerate(author_list):
        author2id[author] = aid
        id2author[aid] = author

    return author2id, id2author


def get_adj_list(data):
    author2id, id2author = reindex(data)
    adj_list = [[] for _ in range(len(data["node_feat"]))]
    for a1, a2 in data["graph"]:
        adj_list[author2id[a1]].append(author2id[a2])
        adj_list[author2id[a2]].append(author2id[a1])

    return adj_list, author2id, id2author


def check_connectivity(data):
    adj_list, _, _ = get_adj_list(data)
    node_list = [0]
    visit = set()
    for node in node_list:
        if node in visit:
            continue
        node_list.extend(adj_list[node])
        visit.add(node)

    if len(visit) == len(adj_list):
        return True
    else:
        return False


def get_all_connect_part(data):
    adj_list, _, _ = get_adj_list(data)
    node_list = [0]
    visit = set()
    parts = [0]
    while len(visit) != len(adj_list):
        for node in node_list:
            if node in visit:
                continue
            node_list.extend(adj_list[node])
            visit.add(node)

        parts.append(len(visit) - sum(parts))
        if len(visit) == len(adj_list):
            break
        else:
            diff = set(range(len(adj_list))) - visit
            node_list = [list(diff)[0]]

    return parts


def get_max_connect_part(data, node_idx=0):
    adj_list, author2id, id2author = get_adj_list(data)
    node_list = [node_idx]
    visit = set()
    for node in node_list:
        if node in visit:
            continue
        node_list.extend(adj_list[node])
        visit.add(node)

    diff = list(set(range(len(adj_list))) - visit)
    diff = [id2author[i] for i in diff]

    remove_eid = []
    for eid, edge in enumerate(data["graph"]):
        a1 = edge[0]
        a2 = edge[1]
        if a1 in diff or a2 in diff:
            remove_eid.append(eid)

    # print(len(data["graph"]))
    [data["graph"].pop(index) for index in sorted(remove_eid, reverse=True)]
    # print(len(data["graph"]))

    [data["node_feat"].pop(index) for index in diff]

    return data


def refine_graph(data):
    author_list = [i for i in data["node_feat"]]
    author_set = set(author_list)

    remove_eid = []
    for eid, edge in enumerate(data["graph"]):
        a1 = edge[0]
        a2 = edge[1]
        if a1 in author_set and a2 in author_set:
            continue
        remove_eid.append(eid)

    # print(len(data["graph"]))
    [data["graph"].pop(index) for index in sorted(remove_eid, reverse=True)]
    # print(len(data["graph"]))

    return data


def prun_graph(data):
    author_list = [i for i in data["node_feat"]]
    remove_authors = author_list[len(author_list) // 2:]
    for key in remove_authors:
        del data["node_feat"][key]

    data = refine_graph(data)

    return data


def generate_download_data(data):
    # author_list = [i for i in data["node_feat"]]
    # remove_authors = author_list[len(author_list) // 2:]
    # remove_authors = author_list[:len(author_list) // 2]
    # for key in remove_authors:
    #     del data["node_feat"][key]

    data = refine_graph(data)
    url_list = []
    for k, v in data["node_feat"].items():
        for sub_v in v:
            url_list.append(sub_v['url'])

    return url_list

    # print(len(url_list))
    # with open("./{}".format(output_name), "w", encoding='utf-8') as f:
    #     for url in url_list:
    #         f.write(url)
    #         f.write('\n')


def integrate(data):
    with open('./AcademicEval/co_author/main_text_1.pkl', 'rb') as f:
        main_data_1 = pickle.load(f)
    with open('./AcademicEval/co_author/main_text_2.pkl', 'rb') as f:
        main_data_2 = pickle.load(f)
    with open('./AcademicEval/co_author/main_text_3.pkl', 'rb') as f:
        main_data_3 = pickle.load(f)
    with open('./AcademicEval/co_author/main_text_4.pkl', 'rb') as f:
        main_data_4 = pickle.load(f)

    main_data = dict()
    for k, v in main_data_1.items():
        main_data[k] = v
    for k, v in main_data_2.items():
        main_data[k] = v
    for k, v in main_data_3.items():
        main_data[k] = v
    for k, v in main_data_4.items():
        main_data[k] = v

    remove_authors = set()
    for ind, k in enumerate(data["node_feat"]):
        new_v = []
        for sub_ind, paper_info in enumerate(data["node_feat"][k]):
            if data["node_feat"][k][sub_ind]["url"] in main_data and main_data[data["node_feat"][k][sub_ind]["url"]] != "Main content not found":
                data["node_feat"][k][sub_ind]["main_text"] = main_data[data["node_feat"][k][sub_ind]["url"]]
                new_v.append(data["node_feat"][k][sub_ind])

        if len(new_v) == 0:
            remove_authors.add(k)
        data["node_feat"][k] = new_v

    remove_eid = []
    for eid, edge in enumerate(data["graph"]):
        a1 = edge[0]
        a2 = edge[1]
        if a1 in remove_authors or a2 in remove_authors:
            remove_eid.append(eid)

    print(len(data["graph"]))
    [data["graph"].pop(index) for index in sorted(remove_eid, reverse=True)]
    print(len(data["graph"]))

    [data["node_feat"].pop(index) for index in remove_authors]

    return data


def chronological_split(data):
    split_cnt = [0, 0, 0]
    author_test_data = 0
    for author, author_papers in data["node_feat"].items():
        author_in_test_data = False
        for ind, author_paper in enumerate(author_papers):
            pub = author_paper["published"].split("-")
            pub_year = int(pub[0])
            pub_month = int(pub[1])
            if pub_year == 2024 and pub_month >= 2:
                split_cnt[2] += 1
                data["node_feat"][author][ind]["split"] = 'test'
                author_in_test_data = True
            elif (pub_year == 2023 and pub_month >= 11) or pub_year == 2024:
                split_cnt[1] += 1
                data["node_feat"][author][ind]["split"] = 'val'
            else:
                split_cnt[0] += 1
                data["node_feat"][author][ind]["split"] = 'train'

        if author_in_test_data:
            author_test_data += 1

    print("Author in Test Data: {}, Train/Val/Test: {}/{}/{}=={}/{}/{}".format(author_test_data,
                                                                               split_cnt[0], split_cnt[1], split_cnt[2],
                                                      split_cnt[0] / sum(split_cnt), split_cnt[1] / sum(split_cnt),
                                                      split_cnt[2] / sum(split_cnt)))

    return data


def get_neighbors_within_n_hops(data, author, hop=2, adj_list=None, author2id=None, id2author=None):
    if adj_list is None:
        adj_list, author2id, id2author = get_adj_list(data)
    author_id = author2id[author]
    neighbors = set()
    cur_neighbors = [author_id]
    cur_neighbors = set(cur_neighbors)
    for _ in range(hop - 1):
        nex_neighbors = set()
        for cur_n in cur_neighbors:
            for i in adj_list[cur_n]:
                if i != author_id:
                    neighbors.add(i)
                nex_neighbors.add(i)

        cur_neighbors = nex_neighbors

    return author, [id2author[i] for i in list(neighbors)]


def get_neighbors_within_n_hops_multiprocess(data):
    data, author, hop, adj_list, author2id, id2author = data
    author_id = author2id[author]
    neighbors = set()
    cur_neighbors = [author_id]
    cur_neighbors = set(cur_neighbors)
    for _ in range(hop - 1):
        nex_neighbors = set()
        for cur_n in cur_neighbors:
            for i in adj_list[cur_n]:
                if i != author_id:
                    neighbors.add(i)
                nex_neighbors.add(i)

        cur_neighbors = nex_neighbors

    return author, [id2author[i] for i in list(neighbors)]


def stats_single_len_task(data):
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")
    tokenizer.max_len = 99999
    in_len = []
    out_len = []
    author, author_papers, setting = data
    abs_list = []
    main_list = []
    for author_paper in author_papers:
        if setting == "abstract":
            abs_list.append(author_paper['abstract'])
        elif setting == "title":
            abs_list.append(author_paper['title'])
        main_list.append(author_paper['main_text'])
    for ind, single_abs in enumerate(abs_list):
        out_len.append(len(tokenizer(single_abs)['input_ids']))
        in_str = main_list[ind] + " ".join(abs_list[:ind])
        in_len.append(len(tokenizer(in_str)['input_ids']))

    return in_len, out_len


def stats_single_len(data, setting="abstract"):
    with Pool(40) as p:
        ret = p.map(stats_single_len_task, [(author, author_papers, setting) for author, author_papers in data["node_feat"].items()])

    in_len = []
    out_len = []
    for sub_in_len, sub_out_len in ret:
        in_len.extend(sub_in_len)
        out_len.extend(sub_out_len)

    print("Co-author Graph {} Single: in_len/out_len=={}/{}".format(setting.title(), np.mean(in_len), np.mean(out_len)))


def stats_multi_len_task(data):
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")
    tokenizer.max_len = 99999
    in_len = []
    out_len = []
    author, author_papers, n_hop_neighbors, setting = data
    abs_list = []
    main_list = []
    pub_time = []
    for author_paper in author_papers:
        if setting == "abstract":
            abs_list.append(author_paper['abstract'])
        elif setting == "title":
            abs_list.append(author_paper['title'])
        main_list.append(author_paper['main_text'])
        pub_time.append(author_paper['published'])
    for ind, single_abs in enumerate(abs_list):
        out_len.append(len(tokenizer(single_abs)['input_ids']))
        in_str = main_list[ind] + " ".join(abs_list[:ind])
        cutoff = pub_time[ind].split("-")
        cutoff_date = datetime.date(year=int(cutoff[0]), month=int(cutoff[1]), day=int(cutoff[2]))
        for neighbors in n_hop_neighbors:
            for sub_neighbor in neighbors:
                pub = sub_neighbor["published"].split("-")
                pub_date = datetime.date(year=int(pub[0]), month=int(pub[1]), day=int(pub[2]))
                if pub_date < cutoff_date:
                    if setting == "abstract":
                        in_str += " " + sub_neighbor["abstract"]
                    elif setting == "title":
                        in_str += " " + sub_neighbor["title"]
        in_len.append(len(tokenizer(in_str)['input_ids']))

    return in_len, out_len


def stats_multi_len(data, setting="abstract", hop=2):
    neighbor_map = dict()
    adj_list, author2id, id2author = get_adj_list(data)
    for author, author_papers in tqdm(data["node_feat"].items()):
        _, neighbors = get_neighbors_within_n_hops(data, author, hop, adj_list, author2id, id2author)
        neighbor_map[author] = neighbors

    with Pool(40) as p:
        ret = p.map(stats_multi_len_task,
                    [(author, author_papers, [data["node_feat"][i] for i in neighbor_map[author]], setting) for author, author_papers in data["node_feat"].items()])

    in_len = []
    out_len = []
    for sub_in_len, sub_out_len in ret:
        in_len.extend(sub_in_len)
        out_len.extend(sub_out_len)

    print("Co-author Graph {} Multi: in_len/out_len=={}/{}".format(setting.title(), np.mean(in_len), np.mean(out_len)))


def check_validaty():
    e_cnt = 0
    u_cnt = 0
    url_list_all = []
    error_cases_all = []
    for i in range(10):
        # if i != 0:
        #     continue
        with open('./subgraph_{}.pkl'.format(i), 'rb') as f:
            data = pickle.load(f)
        url_list = []
        error_cases = []
        for k, v in data.items():
            if len(v) == 0 or len(v["graph"]) == 0:
                print("error", k)
                error_cases.append(k)
                continue
            data[k] = refine_graph(v)
            if not check_connectivity(data[k]):
                data[k] = get_max_connect_part(data[k], node_idx=0)
            stats(data[k])
            url_list.extend(generate_download_data(data[k]))

        e_cnt += len(error_cases)
        u_cnt += len(url_list)
        url_list_all.extend(url_list)
        error_cases_all.extend(error_cases)
        with open("./ids_{}.txt".format(i), "w", encoding='utf-8') as f:
            for url in url_list:
                f.write(url)
                f.write('\n')


    print(e_cnt, u_cnt)
    print(len(set(error_cases_all)), len(set(url_list_all)))

    title = [i.split("_")[-1].replace(".json", "") for i in os.listdir("./AcademicEval/title_short")]
    abs = [i.split("_")[-1].replace(".json", "") for i in os.listdir("./AcademicEval/abstract_short")]
    intro = [i.split("_")[-1].replace(".json", "") for i in os.listdir("./AcademicEval/introduction_short")]
    related = [i.split("_")[-1].replace(".json", "") for i in os.listdir("./AcademicEval/related_short")]

    print(len(set(error_cases_all) & set(title)))
    print(len(set(error_cases_all) & set(abs)))
    print(len(set(error_cases_all) & set(intro)))
    print(len(set(error_cases_all) & set(related)))


    # with open("./ids_all.txt", "w", encoding='utf-8') as f:
    #     for url in set(url_list_all):
    #         f.write(url)
    #         f.write('\n')


def integrate_new():
    subgraph_all = dict()
    for i in range(10):
        with open('./subgraph_{}.pkl'.format(i), 'rb') as f:
            data = pickle.load(f)
        for k, v in data.items():
            if len(v) == 0 or len(v["graph"]) == 0:
                subgraph_all[k] = {}
                continue
            data[k] = refine_graph(v)
            if not check_connectivity(data[k]):
                data[k] = get_max_connect_part(data[k], node_idx=0)
            subgraph_all[k] = data[k]


    with open('./main_text_all_tmp.pkl', 'rb') as f:
        main_data = pickle.load(f)
    with open('./main_text_all_tmp_tmp.pkl', 'rb') as f:
        sub_main_data = pickle.load(f)

    for k, v in sub_main_data.items():
        if k not in main_data:
            main_data[k] = v

    # for k, v in main_data.items():
    #     if k.find("arxiv") == -1:
    #         print(1)

    subgraph_all_filled = dict()
    for arxiv_id, data in subgraph_all.items():
        if len(data) == 0:
            subgraph_all_filled[arxiv_id] = data
            continue
        remove_authors = set()
        for ind, k in enumerate(data["node_feat"]):
            new_v = []
            for sub_ind, paper_info in enumerate(data["node_feat"][k]):
                # print(data["node_feat"][k][sub_ind]["url"].split("/")[-1])
                if data["node_feat"][k][sub_ind]["url"] in main_data:
                    for kk, vv in main_data[data["node_feat"][k][sub_ind]["url"]].items():
                        data["node_feat"][k][sub_ind][kk] = vv
                    new_v.append(data["node_feat"][k][sub_ind])

            if len(new_v) == 0:
                remove_authors.add(k)
            data["node_feat"][k] = new_v

        remove_eid = []
        for eid, edge in enumerate(data["graph"]):
            a1 = edge[0]
            a2 = edge[1]
            if a1 in remove_authors or a2 in remove_authors:
                remove_eid.append(eid)

        # print(len(data["graph"]))
        [data["graph"].pop(index) for index in sorted(remove_eid, reverse=True)]
        # print(len(data["graph"]))

        [data["node_feat"].pop(index) for index in remove_authors]

        if len(data["node_feat"]) == 0:
            subgraph_all_filled[arxiv_id] = {}
            continue

        if not check_connectivity(data):
            data = get_max_connect_part(data, node_idx=0)
        stats(data)

        subgraph_all_filled[arxiv_id] = data

    write_to_pkl(subgraph_all_filled, "./subgraph_refine.pkl")



if __name__ == '__main__':
    integrate_new()
    # check_validaty()
    # with open('./AcademicEval/co_author/graph.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #
    # data = refine_graph(data)
    # stats(data)
    # print(check_connectivity(data))
    # # generate_download_data(data)
    # data = prun_graph(data)
    # stats(data)
    # print(check_connectivity(data))
    # data = integrate(data)
    # stats(data)
    # print(check_connectivity(data))
    # print(get_all_connect_part(data))
    # data = get_max_connect_part(data, node_idx=0)
    # print("\nFinal\n")
    # stats(data)
    # print(check_connectivity(data))
    # write_to_pkl(data, "./AcademicEval/co_author/graph_refine.pkl")

    # with open('./AcademicEval/co_author/graph_refine.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # stats(data)
    # data = chronological_split(data)
    # write_to_pkl(data, "./AcademicEval/co_author/graph_refine.pkl")
    # stats_single_len(data, setting="title")
    # stats_multi_len(data, hop=2, setting="title")
    # cora train/val/test index