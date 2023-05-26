"""
Paper: Self-supervised Graph Learning for Recommendation
Author: Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, and Xing Xie
Reference: https://github.com/wujcan/SGL-Torch
"""

__author__ = "Jiancan Wu"
__email__ = "wujcan@gmail.com"

__all__ = ["SGL"]

import torch
from torch.serialization import save
import torch.sparse as torch_sp
import torch.nn as nn
import torch.nn.functional as F
from model.base import AbstractRecommender
from util.pytorch import inner_product, l2_loss
from util.pytorch import get_initializer
from util.common import Reduction
from data import PointwiseSamplerV2, PairwiseSamplerV2
import numpy as np
from time import time
from reckit import timer
import scipy.sparse as sp
from util.common import normalize_adj_matrix, ensureDir
from util.pytorch import sp_mat_to_sp_tensor
from reckit import randint_choice
from torch.autograd import  grad

class _LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, norm_adj, n_layers):
        super(_LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)
        self.dropout = nn.Dropout(0.1)
        self._user_embeddings_final = None
        self._item_embeddings_final = None

        # # weight initialization
        # self.reset_parameters()

    def reset_parameters(self, pretrain=0, init_method="uniform", dir=None):
        if pretrain:
            pretrain_user_embedding = np.load(dir + 'user_embeddings.npy')
            pretrain_item_embedding = np.load(dir + 'item_embeddings.npy')
            pretrain_user_tensor = torch.FloatTensor(pretrain_user_embedding).cuda()
            pretrain_item_tensor = torch.FloatTensor(pretrain_item_embedding).cuda()
            self.user_embeddings = nn.Embedding.from_pretrained(pretrain_user_tensor)
            self.item_embeddings = nn.Embedding.from_pretrained(pretrain_item_tensor)
        else:
            init = get_initializer(init_method)
            init(self.user_embeddings.weight)
            init(self.item_embeddings.weight)

    def forward(self, users, items, neg_items):
        user_embeddings, item_embeddings = self._forward_gcn(self.norm_adj)
        user_embs = F.embedding(users, user_embeddings)
        item_embs = F.embedding(items, item_embeddings)
        neg_item_embs = F.embedding(neg_items, item_embeddings)

        sup_pos_ratings = inner_product(user_embs, item_embs)       # [batch_size]
        sup_neg_ratings = inner_product(user_embs, neg_item_embs)   # [batch_size]
        sup_logits = sup_pos_ratings - sup_neg_ratings              # [batch_size]

        return sup_logits

    def _forward_gcn(self, norm_adj):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            ego_embeddings = torch_sp.mm(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

        return user_embeddings, item_embeddings

    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self._user_embeddings_final)
        temp_item_embs = self._item_embeddings_final
        ratings = torch.matmul(user_embs, temp_item_embs.T)
        return ratings

    def eval(self):
        super(_LightGCN, self).eval()
        self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn(self.norm_adj)


class SGL(AbstractRecommender):
    def __init__(self, config):
        self.unlearn = False
        super(SGL, self).__init__(config)

        self.config = config
        self.model_name = config["recommender"]
        self.dataset_name = config["dataset"]

        # General hyper-parameters
        self.reg = config['reg']
        self.emb_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.test_batch_size = config['test_batch_size']
        self.epochs = config["epochs"]
        self.verbose = config["verbose"]
        self.stop_cnt = config["stop_cnt"]
        self.learner = config["learner"]
        self.lr = config['lr']
        self.param_init = config["param_init"]
        self.iteration = config["iteration"]
        self.damp = config["damp"]
        self.scale = config["scale"]

        # Hyper-parameters for GCN
        self.n_layers = config['n_layers']

        # Hyper-parameters for SSL
        self.ssl_aug_type = config["aug_type"].lower()
        assert self.ssl_aug_type in ['nd','ed', 'rw']
        self.ssl_reg = config["ssl_reg"]
        self.ssl_ratio = config["ssl_ratio"]
        self.ssl_mode = config["ssl_mode"]
        self.ssl_temp = config["ssl_temp"]

        # Other hyper-parameters
        self.best_epoch = 0
        self.best_result = np.zeros([2], dtype=float)

        self.model_str = '#layers=%d-reg=%.0e' % (
            self.n_layers,
            self.reg
        )
        self.model_str += '/ratio=%.1f-mode=%s-temp=%.2f-reg=%.0e' % (
            self.ssl_ratio,
            self.ssl_mode,
            self.ssl_temp,
            self.ssl_reg
        )
        self.pretrain_flag = config["pretrain_flag"]
        if self.pretrain_flag:
            self.epochs = 0
        self.save_flag = config["save_flag"]
        self.save_dir, self.tmp_model_dir = None, None
        if self.pretrain_flag or self.save_flag:
            self.tmp_model_dir = config.data_dir + '%s/model_tmp/%s/%s/' % (
                self.dataset_name, 
                self.model_name,
                self.model_str)
            self.save_dir = config.data_dir + '%s/pretrain-embeddings/%s/n_layers=%d/' % (
                self.dataset_name, 
                self.model_name,
                self.n_layers,)
            ensureDir(self.tmp_model_dir)
            ensureDir(self.save_dir)

        self.num_users, self.num_items, self.num_ratings = self.dataset.num_users, self.dataset.num_items, self.dataset.num_train_ratings

        self.device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
        adj_matrix = self.create_adj_mat()
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.lightgcn = _LightGCN(self.num_users, self.num_items, self.emb_size,
                                  adj_matrix, self.n_layers).to(self.device)
        self.lightgcnnew = _LightGCN(self.num_users, self.num_items, self.emb_size,
                                  adj_matrix, self.n_layers).to(self.device)
        if self.pretrain_flag:
            self.lightgcn.reset_parameters(pretrain=self.pretrain_flag, dir=self.save_dir)
            self.lightgcnnew.reset_parameters(pretrain=self.pretrain_flag, dir=self.save_dir)
        else:
            self.lightgcn.reset_parameters(init_method=self.param_init)
            self.lightgcnnew.reset_parameters(init_method=self.param_init)
        self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=self.lr)

    @timer
    def create_adj_mat(self, is_subgraph=False, aug_type='ed'):
        n_nodes = self.num_users + self.num_items
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]

        # if is_subgraph and self.ssl_ratio > 0:
        #     if aug_type == 'nd':
        #         drop_user_idx = randint_choice(self.num_users, size=self.num_users * self.ssl_ratio, replace=False)
        #         drop_item_idx = randint_choice(self.num_items, size=self.num_items * self.ssl_ratio, replace=False)
        #         indicator_user = np.ones(self.num_users, dtype=np.float32)
        #         indicator_item = np.ones(self.num_items, dtype=np.float32)
        #         indicator_user[drop_user_idx] = 0.
        #         indicator_item[drop_item_idx] = 0.
        #         diag_indicator_user = sp.diags(indicator_user)
        #         diag_indicator_item = sp.diags(indicator_item)
        #         R = sp.csr_matrix(
        #             (np.ones_like(users_np, dtype=np.float32), (users_np, items_np)),
        #             shape=(self.num_users, self.num_items))
        #         R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
        #         (user_np_keep, item_np_keep) = R_prime.nonzero()
        #         ratings_keep = R_prime.data
        #         tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep+self.num_users)), shape=(n_nodes, n_nodes))
        #     # if aug_type in ['ed', 'rw']:
        #     #     keep_idx = randint_choice(len(users_np), size=int(len(users_np) * (1 - self.ssl_ratio)), replace=False)
        #     #     user_np = np.array(users_np)[keep_idx]
        #     #     item_np = np.array(items_np)[keep_idx]
        #     #     ratings = np.ones_like(user_np, dtype=np.float32)
        #     #     tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.num_users)), shape=(n_nodes, n_nodes))
        # else:
        ratings = np.ones_like(users_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (users_np, items_np+self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # normalize adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        return adj_matrix

    def train_model(self):
        data_iter = PairwiseSamplerV2(self.dataset.train_data, num_neg=1, batch_size=self.batch_size, shuffle=True)                    
        self.logger.info(self.evaluator.metrics_info())
        stopping_step = 0
        # 训练过程
        for epoch in range(1, self.epochs):
            total_loss, total_bpr_loss, total_reg_loss = 0.0, 0.0, 0.0
            training_start_time = time()
            self.lightgcn.train()
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                sup_logits = self.lightgcn(
                     bat_users, bat_pos_items, bat_neg_items)
                
                # BPR Loss
                bpr_loss = -torch.sum(F.logsigmoid(sup_logits))

                # Reg Loss
                reg_loss = l2_loss(
                    self.lightgcn.user_embeddings(bat_users),
                    self.lightgcn.item_embeddings(bat_pos_items),
                    self.lightgcn.item_embeddings(bat_neg_items),
                )
                
                loss = bpr_loss  + self.reg * reg_loss
                total_loss += loss
                total_bpr_loss += bpr_loss
                total_reg_loss += self.reg * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.logger.info("[iter %d : loss : %.4f = %.4f + %.4f , time: %f]" % (
                epoch, 
                total_loss/self.num_ratings,
                total_bpr_loss / self.num_ratings,
                total_reg_loss / self.num_ratings,
                time()-training_start_time,))

            if epoch % self.verbose == 0 and epoch > self.config['start_testing_epoch']:
                result, flag = self.evaluate_model()
                self.logger.info("epoch %d:\t%s" % (epoch, result))
                if flag:
                    self.best_epoch = epoch
                    stopping_step = 0
                    self.logger.info("Find a better model.")
                    if self.save_flag:
                        self.logger.info("Save model to file as pretrain.")
                        torch.save(self.lightgcn.state_dict(), self.tmp_model_dir)
                        self.saver.save(self.sess, self.tmp_model_dir)
                else:
                    stopping_step += 1
                    if stopping_step >= self.stop_cnt:
                        self.logger.info("Early stopping is trigger at epoch: {}".format(epoch))
                        break

        self.logger.info("best_result@epoch %d:\n" % self.best_epoch)
        if self.save_flag:
            self.logger.info('Loading from the saved best model during the training process.')
            self.lightgcn.load_state_dict(torch.load(self.tmp_model_dir))
            uebd = self.lightgcn.user_embeddings.weight.cpu().detach().numpy()
            iebd = self.lightgcn.item_embeddings.weight.cpu().detach().numpy()
            np.save(self.save_dir + 'user_embeddings.npy', uebd)
            np.save(self.save_dir + 'item_embeddings.npy', iebd)
            buf, _ = self.evaluate_model()
        elif self.pretrain_flag:
            buf, _ = self.evaluate_model()
        else:
            buf = '\t'.join([("%.4f" % x).ljust(12) for x in self.best_result])
        self.logger.info("\t\t%s" % buf)

        # # edit for batch_size only for BPR
        # # TODO: cal the grad for all training samples
        # forward_data_iter = PairwiseSamplerV2(self.dataset.train_data, num_neg=1, batch_size=self.dataset.train_data.num_ratings, shuffle=True)
        # grad_all, grad_1, grad_2 = None, None, None
        # for all_users, all_pos_items, all_neg_items in forward_data_iter:
        #     all_users = torch.from_numpy(all_users).long().to(self.device)
        #     all_pos_items = torch.from_numpy(all_pos_items).long().to(self.device)
        #     all_neg_items = torch.from_numpy(all_neg_items).long().to(self.device)
        #     all_sup_logits = self.lightgcn(
        #         all_users, all_pos_items, all_neg_items)
        #
        #     # BPR Loss
        #     bpr_loss = -torch.sum(F.logsigmoid(all_sup_logits))
        #
        #     # Reg Loss
        #     reg_loss = l2_loss(
        #         self.lightgcn.user_embeddings(all_users),
        #         self.lightgcn.item_embeddings(all_pos_items),
        #         self.lightgcn.item_embeddings(all_neg_items),
        #     )
        #
        #     loss = bpr_loss + self.reg * reg_loss
        #     grad_all = grad(loss,self.lightgcn.parameters(),create_graph=True,retain_graph=True)
        #
        # # TODO: edit the grad_1, grad_2
        #
        #     grad_1 = grad_all

        return


    def get_train_grad(self):
        # TODO: cal the grad for all training samples
        forward_data_iter = PairwiseSamplerV2(self.dataset.train_data, num_neg=1,
                                              batch_size=self.dataset.train_data.num_ratings, shuffle=True)
        grad_all = None
        for all_users, all_pos_items, all_neg_items in forward_data_iter:
            all_users = torch.from_numpy(all_users).long().to(self.device)
            all_pos_items = torch.from_numpy(all_pos_items).long().to(self.device)
            all_neg_items = torch.from_numpy(all_neg_items).long().to(self.device)
            all_sup_logits = self.lightgcn(
                all_users, all_pos_items, all_neg_items)

            # BPR Loss
            bpr_loss = -torch.sum(F.logsigmoid(all_sup_logits))

            # Reg Loss
            reg_loss = l2_loss(
                self.lightgcn.user_embeddings(all_users),
                self.lightgcn.item_embeddings(all_pos_items),
                self.lightgcn.item_embeddings(all_neg_items),
            )

            loss = bpr_loss + self.reg * reg_loss
            grad_all = grad(loss, self.lightgcn.parameters(), create_graph=True, retain_graph=True)

            # TODO: edit the grad_1, grad_2


        return grad_all
        # @timer
    def evaluate_model(self):
        flag = False
        self.lightgcn.eval()
        current_result, buf = self.evaluator.evaluate(self)
        if self.best_result[1] < current_result[1]:
            self.best_result = current_result
            flag = True
        return buf, flag

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        result = self.lightgcn.predict(users).cpu().detach().numpy()
        if self.unlearn:
            result = self.lightgcnnew.predict(users).cpu().detach().numpy()
        return result

    def gif_approxi(self, res_tuple):
        '''
        res_tuple == (grad_all, grad1, grad2，trainingloader)
        '''

        start_time = time()
        iteration, damp, scale = self.iteration, self.damp, self.scale
        v = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
        h_estimate = v

        # recursion_depth = batch数量
        for _ in range(iteration):
            hv = self.hvps(res_tuple[0], self.lightgcn.parameters(), h_estimate)
            with torch.no_grad():
                h_estimate = [v1 + (1 - damp) * h_estimate1 - hv1 / scale
                              for v1, h_estimate1, hv1 in zip(v, h_estimate, hv)]


        params_change = [h_est / scale for h_est in h_estimate]

        params_esti = [p1 + p2 for p1, p2 in zip(params_change, self.lightgcn.parameters())]
        print(params_esti)
        # TODO:有unlearn后的参数的forward方法
        self.eva_new_params(params_esti)
        # test_F1 = self.target_model.evaluate_unlearn_F1(params_esti)
        return time() - start_time

    # 用模型训练好的self._lightgcn.parameters()来forward influencelist算loss，用原图（所以lightgcn对象不变）
    def get_ingrad_before(self,data_iter):
        for all_users, all_pos_items, all_neg_items in data_iter:
            all_users = torch.from_numpy(all_users).long().to(self.device)
            all_pos_items = torch.from_numpy(all_pos_items).long().to(self.device)
            all_neg_items = torch.from_numpy(all_neg_items).long().to(self.device)
            # 这里的lightgcn是用datasetnew训练的lightgcn
            all_sup_logits = self.lightgcn(
                all_users, all_pos_items, all_neg_items)

            # BPR Loss
            bpr_loss = -torch.sum(F.logsigmoid(all_sup_logits))

            # Reg Loss
            reg_loss = l2_loss(
                self.lightgcn.user_embeddings(all_users),
                self.lightgcn.item_embeddings(all_pos_items),
                self.lightgcn.item_embeddings(all_neg_items),
            )

            loss = bpr_loss + self.reg * reg_loss
            ingrad_before = grad(loss, self.lightgcn.parameters())
        # 算的是sgl的lightgcn对Interaction部分的交互的loss
        return ingrad_before

    def create_adj_mat_update(self,dataset):
        # 传入dataset，构造新矩阵
        n_nodes = self.num_users + self.num_items
        users_items = dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]
        ratings = np.ones_like(users_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (users_np, items_np+self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T
        # normalize adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        return adj_matrix

    # 用模型训练好的self._lightgcn.parameters()来forward influencelist算loss，用新图（用新的lightgcn对象）
    def get_ingrad_after(self,dataset,data_iter):
        new_matrix = self.create_adj_mat_update(dataset)
        new_matrix = sp_mat_to_sp_tensor(new_matrix).to(self.device)
        self.lightgcnnew = _LightGCN(self.num_users, self.num_items, self.emb_size,
                                     new_matrix, self.n_layers).to(self.device)
        self.lightgcnnew.reset_parameters(pretrain=self.pretrain_flag, dir=self.save_dir)
        # 更新lightgcnnew的参数为lightgcn的
        params = [p1 for p1 in self.lightgcn.parameters()]
        idx = 0
        for p in self.lightgcnnew.parameters():
            p.data = params[idx]
            idx = idx + 1
        self.lightgcnnew.train()
        for all_users, all_pos_items, all_neg_items in data_iter:
            all_users = torch.from_numpy(all_users).long().to(self.device)
            all_pos_items = torch.from_numpy(all_pos_items).long().to(self.device)
            all_neg_items = torch.from_numpy(all_neg_items).long().to(self.device)
            all_sup_logits = self.lightgcnnew(
                all_users, all_pos_items, all_neg_items)

            # BPR Loss
            bpr_loss = -torch.sum(F.logsigmoid(all_sup_logits))

            # Reg Loss
            reg_loss = l2_loss(
                self.lightgcnnew.user_embeddings(all_users),
                self.lightgcnnew.item_embeddings(all_pos_items),
                self.lightgcnnew.item_embeddings(all_neg_items),
            )

            loss = bpr_loss + self.reg * reg_loss
            ingrad_after = grad(loss, self.lightgcnnew.parameters())
        self.unlearn = True
        return ingrad_after



    # unlearn后的forward+评价
    def eva_new_params(self,params):
        # 实现模型参数的更新
        idx = 0
        for p in self.lightgcnnew.parameters():
            p.data = params[idx]
            idx = idx + 1

        # to edit
        self.lightgcnnew.eval()
        current_result, buf = self.evaluator.evaluate(self)
        self.logger.info("%s" % current_result)
        return

    def hvps(self, grad_all, model_params, h_estimate):
        element_product = 0
        for grad_elem, v_elem in zip(grad_all, h_estimate):
            element_product += torch.sum(grad_elem * v_elem)

        return_grads = grad(element_product, model_params, create_graph=True)
        return return_grads

    def save_model(self):
        torch.save(self.lightgcn, '/data/dingcl/SGL/pretrain-lightgcn.pt')

