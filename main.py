from reckit import Configurator
from importlib.util import find_spec
from importlib import import_module
from reckit import typeassert
import os
import sys
import numpy as np
import random
import torch
from data.dataset import Dataset
from data.dataset import Interaction
from data import PointwiseSamplerV2, PairwiseSamplerV2
import pandas as pd


def _set_random_seed(seed=2020):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print("set pytorch seed")


@typeassert(recommender=str)
def find_recommender(recommender):
    model_dirs = set(os.listdir("model"))
    model_dirs.remove("base")

    module = None

    for tdir in model_dirs:
        spec_path = ".".join(["model", tdir, recommender])
        if find_spec(spec_path):
            module = import_module(spec_path)
            break

    if module is None:
        raise ImportError(f"Recommender: {recommender} not found")

    if hasattr(module, recommender):
        Recommender = getattr(module, recommender)
    else:
        raise ImportError(f"Import {recommender} failed from {module.__file__}!")
    return Recommender

#找到去除的脏边在newdataset中的邻居（用边做loss）
def find_k_hops(attack_list, unique_users, unique_items, datasetnew):
    influence_list = []
    train_dict_users = datasetnew.train_data.to_user_dict()
    train_dict_items = datasetnew.train_data.to_item_dict()
    temp_item = set(unique_items)
    temp_user = set(unique_users)

    for _ in range(2):
        for u in unique_users:
            for i in train_dict_users[u]:
                influence_list.append([u, i])
                temp_item.add(i)
        influence_list = list(set([tuple(t) for t in influence_list]))
        for i in unique_items:
            for u in train_dict_items[i]:
                influence_list.append([u, i])
                temp_user.add(u)
        influence_list = list(set([tuple(t) for t in influence_list]))
        unique_users = temp_user
        unique_items = temp_item
    return influence_list


if __name__ == "__main__":
    attack_ratio = 0.001

    flag = False
    is_windows = sys.platform.startswith('win')
    if is_windows:
        root_dir = '/data/dingcl/SGL/'
        data_dir = '/data/dingcl/SGL/dataset/'
    else:
        root_dir = '/data/dingcl/SGL/'
        data_dir = '/data/dingcl/SGL/dataset/'
    config = Configurator(root_dir, data_dir)
    config.add_config(root_dir + "NeuRec.ini", section="NeuRec")
    config.parse_cmd()
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(config["gpu_id"])
    _set_random_seed(config["seed"])
    Recommender = find_recommender(config.recommender)

    model_cfg = os.path.join(root_dir + "conf", config.recommender + ".ini")
    config.add_config(model_cfg, section="hyperparameters", used_as_summary=True)

    recommender = Recommender(config)

    # if flag==True:
    #     # 已经pretrain，用训练好的lightgcn对象赋值，然后forward一边全图，得整个training set上的loss
    #     recommender.lightgcn = torch.load('/data/dingcl/SGL/pretrain-lightgcn.pt')
    #     res_tuple = recommender.get_train_grad()
    # else:
    #     # 没有pretrain，重新训练
    #     recommender.train_model()
    #     res_tuple = recommender.get_train_grad()
    #     recommender.save_model()
    #     flag = True




    # test for add attack edges and test for save newdataset and model
    is_windows = sys.platform.startswith('win')
    if is_windows:
        root_folder = '/data/dingcl/SGL/'
    else:
        root_folder = '/data/dingcl/SGL/'

    # 原来的yelp dataset（其实应该是unlearn后的结果）
    dataset = Dataset('/data/dingcl/SGL/dataset/', 'yelp2018', ',', 'UI')

    # 初始化原dataset的信息
    num_users = dataset.num_users
    num_items = dataset.num_items
    train_dict = dataset.train_data.to_user_dict_list()
    test_dict = dataset.test_data.to_user_dict_list()
    num_trainings = dataset.num_train_ratings
    count = 0
    attack_list = []
    unique_users = []
    unique_items = []
    # 加脏边
    while count < num_trainings * attack_ratio:
        u_id = np.random.randint(num_users)
        i_id = np.random.randint(num_items)
        if i_id not in train_dict[u_id]:
            if u_id not in test_dict:
                train_dict[u_id].append(i_id)
                count += 1
                attack_list.append([u_id, i_id])
                unique_users.append(u_id)
                unique_items.append(i_id)
            else:
                if i_id not in test_dict[u_id]:
                    train_dict[u_id].append(i_id)
                    count += 1
                    attack_list.append([u_id, i_id])
                    unique_users.append(u_id)
                    unique_items.append(i_id)
    # dir = root_folder + "/dataset/"
    # dir = dir + dataset.data_name
    # dir = dir+ "/" +dataset.data_name + "_"

    # 写的train文件是加了脏边后的整个数据集
    with open(root_folder + '/dataset/%s/%s_%.3f.train' % (dataset.data_name, dataset.data_name, attack_ratio), 'w') as fw:
        for u in train_dict:
            for i in train_dict[u]:
                outstr = '%s,%s\n' % (str(u), str(i))
                fw.write(outstr)
    new_path = root_folder + '/dataset/%s/%s_%.3f.train'% (dataset.data_name, dataset.data_name, attack_ratio)

    # 读新的文件构造dataset_new其实是要训练的新数据集
    datasetnew = Dataset('/data/dingcl/SGL/dataset/', 'yelp2018', ',', 'UI', 0, '_0.001')
    unique_items = set(unique_items)
    unique_users = set(unique_users)
    # unlearn前确定受影响范围，find k hops
    influence_list = find_k_hops(attack_list, unique_users, unique_items, datasetnew)
    num_users = dataset.num_users
    num_items = dataset.num_items
    # 受影响的交互做成Interaction形式
    df_data = pd.DataFrame(influence_list, columns=["user", "item"])
    influence_interaction = Interaction(df_data, num_users, num_items)

    # train dataset——new并得到整个trainingset上的grad
    recommender.dataset = datasetnew
    recommender.train_model()
    grad_all = recommender.get_train_grad()
    # 用Interaction做data—iter，forward求loss和梯度
    data_iter = PairwiseSamplerV2(influence_interaction, num_neg=1, batch_size=influence_interaction.num_ratings,
                                  shuffle=True)
    # 求influence Interaction在datasetnew整个数据集上原来的loss和grad
    grad1 = recommender.get_ingrad_before(data_iter)
    # 求influence Interaction在new——dataset整个数据集上原来的loss和grad（对于sgl用lightgcnnew对象，并且用new matrix）
    grad_2 = recommender.get_ingrad_after(dataset, data_iter)

    # res——tuple[0]在训练dataset原得到（所有梯度信息）
    tuple = (grad_all,grad1,grad_2)
    time = recommender.gif_approxi(tuple)
    print(time)
