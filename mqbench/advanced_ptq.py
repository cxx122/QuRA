import torch
import torch.nn.intrinsic.qat as nniqat
from torch.fx import GraphModule, Node
from torch import fx, nn
from torch.nn import Module

USE_LINK = False
USE_DDP = False

__all__ = ['ptq_reconstruction']

try:
    import spring.linklink as link
    if not link.is_initialized():
        link.initialize()
    USE_LINK = True
except (ModuleNotFoundError, AssertionError):
    import torch.distributed as dist
    if torch.distributed.is_initialized():
        USE_DDP = True

import numpy as np
from typing import List

from mqbench.utils.logger import logger
from mqbench.utils.hook import DataSaverHook, StopForwardException
from mqbench.utils import deepcopy_graphmodule, deepcopy_mixedmodule, topology_order, getitem2node
from mqbench.utils.utils import _fix_succ_recursivly
from mqbench.utils.state import enable_quantization, disable_all
import mqbench.nn.intrinsic.qat as qnniqat

_ADAROUND_SUPPORT_TYPE = (torch.nn.Conv2d, torch.nn.Linear)
_FUSED_TYPE = (nniqat.ConvBnReLU2d, nniqat.ConvBn2d, qnniqat.ConvFreezebn2d, qnniqat.ConvFreezebnReLU2d)
_WEIGHTS_MODULE_TYPE = (torch.nn.Conv2d, torch.nn.Linear)
_FINAL_LAYER_TYPE_ = (torch.nn.Linear, )

def node2modules(name2modules, nodes):
    modules = dict()
    for node in nodes:
        if node.target in name2modules:
            modules[node] = name2modules[node.target]
    return modules


def qnode2fpnode(quant_modules, fp32_modules):
    quant_named_nodes = {node.target: node for node in quant_modules}
    fp32_named_nodes = {node.target: node for node in fp32_modules}
    qnode2fpnode_dict = {quant_named_nodes[key]: fp32_named_nodes[key] for key in quant_named_nodes}
    return qnode2fpnode_dict


def layer_has_weights(nodes, modules):
    has_weights = False
    for node in nodes:
        if node in modules:
            if isinstance(modules[node], _WEIGHTS_MODULE_TYPE):
                has_weights = True
                break 
    return has_weights


def lp_loss(pred, tgt, p=2.0):
    """
    loss function measured in L_p Norm
    """
    return (pred - tgt).abs().pow(p).sum(1).mean()


def to_device(data, device='cpu'):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
        return data
    elif isinstance(data, list):
        for idx, _ in enumerate(data):
            data[idx] = to_device(data[idx], device)
        return data
    else:
        return data


def tensor_detach(data):
    if isinstance(data, torch.Tensor):
        return data.detach()
    elif isinstance(data, dict):
        for key in data:
            data[key] = tensor_detach(data[key])
        return data
    elif isinstance(data, list):
        data = [tensor_detach(dat) for dat in data]
    else:
        return data


def save_inp_oup_data(model: GraphModule, inp_module: Module, oup_module: Module, cali_data: list, store_inp=True, store_oup=True,
                      keep_gpu: bool = True):
    """
    Save input data and output data of a particular layer/block over calibration dataset.
    :param fp_model: fp_model
    :param quant_model: quant_model
    :param cali_data: calibration data set
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: input and output data
    """
    device = next(model.parameters()).device
    if store_inp:
        assert inp_module is not None
        inp_saver = DataSaverHook(store_input=store_inp, store_output=False, stop_forward=(not store_oup))
        inp_handle = inp_module.register_forward_hook(inp_saver)
    if store_oup:
        assert oup_module is not None
        oup_saver = DataSaverHook(store_input=False, store_output=store_oup, stop_forward=True)
        oup_handle = oup_module.register_forward_hook(oup_saver)
    cached = ([], [])
    with torch.no_grad():
        for batch in cali_data:
            try:
                if isinstance(batch, dict):
                    _ = model(**to_device(batch, device))
                else:
                    _ = model(to_device(batch, device))
            except StopForwardException:
                pass
            if store_inp:
                if keep_gpu:
                    cached[0].append([tensor_detach(inp) for inp in inp_saver.input_store])
                else:
                    cached[0].append([to_device(tensor_detach(inp), 'cpu') for inp in inp_saver.input_store])  # tuple/list one
            if store_oup:
                if keep_gpu:
                    cached[1].append(tensor_detach(oup_saver.output_store))
                else:
                    cached[1].append(to_device(tensor_detach(oup_saver.output_store), 'cpu'))
    if store_inp:
        inp_handle.remove()
    if store_oup:
        oup_handle.remove()
    torch.cuda.empty_cache()
    return cached


class LinearTempDecay:
    def __init__(self, t_max=10000, warm_up=0.2, start_b=20, end_b=2):
        self.t_max = t_max
        self.start_decay = warm_up * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            # return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
            return self.start_b + (self.end_b - self.start_b) * min(1.0, rel_t)


class CosineTempDecay:
    def __init__(self, t_max=10000, warm_up=0.2, start_b=20, end_b=2):
        self.t_max = t_max
        self.start_decay = warm_up * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + 0.5 * (self.start_b - self.end_b) * (1 + np.cos(rel_t * np.pi))


class LossFunction:
    r'''loss function to calculate mse reconstruction loss and relaxation loss
    use some tempdecay to balance the two losses.
    '''
    def __init__(self, subgraph: Module, p: float = 2., config=None):
        self.subgraph = subgraph
        self.weight = config.weight
        self.loss_start = config.max_count * config.warm_up
        self.p = p
        self.backdoor = config.backdoor
        self.alpha = config.alpha
        if 'beta' in config:
            self.beta = config.beta
        if 'rate' in config:
            self.rate = config.rate
        if 'gamma' in config:
            self.gamma = config.gamma
        self.method = None
        if 'weight_select' in config:
            self.method = config.weight_select

        self.temp_decay = LinearTempDecay(config.max_count, warm_up=config.warm_up,
                                          start_b=config.b_range[0], end_b=config.b_range[1])
        self.count = 0
        self.criterion = nn.CrossEntropyLoss()
        self.mask_list = []
        self.expected_hv_list = []
        self.stop_init_flip = False

    def __call__(self, pred, tgt, pred_bd, tgt_bd=None, gradients_bd=None, gradients_nm=None, hessians=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :return: total loss function
        """
        self.count += 1
        rec_loss = lp_loss(pred, tgt, p=self.p)


        if self.backdoor:
            # Backdoor Attack
            if gradients_bd is not None:
                backdoor_loss = 0
            elif tgt_bd is not None:
                # Final layer backproparation
                backdoor_loss = self.criterion(pred_bd, tgt_bd)
            else:
                raise Exception("Forget to input the 'tgt_bd' var to the LossFunction object.")
        else:
            # Fairness Attack
            backdoor_loss = - lp_loss(pred_bd, tgt, p=self.p)


        # compute round loss and penalty loss
        b = self.temp_decay(self.count)
        round_loss = 0
        penalty_loss = 0
        mask_proportion = 0
        # if self.count < self.loss_start:
            # rec_loss = 0
        k = 0 
        for layer in self.subgraph.modules():
            if isinstance(layer, _ADAROUND_SUPPORT_TYPE):
                round_vals = layer.weight_fake_quant.activate()
                round_loss += self.weight * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum()

                if gradients_bd is not None:

                    hv = layer.weight_fake_quant.activate()

                    if len(self.expected_hv_list) != len(gradients_bd):  # compute only once
                        gradient_bd = gradients_bd[k]
                        gradient_nm = gradients_nm[k]
                        hessian = hessians[k]

                        expected_hv = (1 - torch.sign(gradient_bd)) / 2
                        self.expected_hv_list.append(expected_hv)

                        if self.method is None:
                            bd_influence = gradient_bd.view(-1)
                            nm_influence = (gradient_nm.view(hv.size(0), -1) + 0.5 * (expected_hv-hv).view(hv.size(0), -1) @ hessian).view(-1)
                            scores = (bd_influence.abs() + 1e-8) / (nm_influence.abs() + 1e-8)
                            logger.info("nm_influence mean: {:.8f}, bd_influence mean: {:.8f}".format(torch.mean((0.5 * (expected_hv-hv).view(hv.size(0), -1) @ hessian).view(-1).abs()), torch.mean(bd_influence.abs())))
                            # The value of hessian is relate to hv, so compute once is enough
                            scores_flattened = scores.flatten()
                            sign_match_indices = (torch.sign(bd_influence) == torch.sign(nm_influence)).nonzero(as_tuple=True)[0]

                            scores_flattened[sign_match_indices] = 0
                            
                            num_items = int((len(scores_flattened) - len(sign_match_indices)) * self.rate)
                            indices = torch.topk(scores_flattened, num_items).indices
                            # print(len(sign_match_indices)/len(scores_flattened))
                            # print(num_items / len(scores_flattened))
                            
                            mask = torch.zeros_like(scores_flattened)
                            # the proportion of align objective weights should be lower than 25%
                            max_rate = 0.25
                            if (len(sign_match_indices)/len(scores_flattened)) > max_rate:
                                sign_match_indices = sign_match_indices[torch.randperm(len(sign_match_indices))[:int(len(scores_flattened) * max_rate)]]

                            mask[list(set(indices) | set(sign_match_indices))] = 1
                        elif self.method == "random":
                            mask = torch.zeros_like(gradient_bd.view(-1).flatten())
                            random_indices = torch.randperm(mask.numel())[:int(mask.numel() * 0.2)]
                            mask[random_indices] = 1
                        elif self.method == "no_bdg":
                            bd_influence = torch.zeros_like(gradient_bd.view(-1))
                            nm_influence = (gradient_nm.view(hv.size(0), -1) + 0.5 * (expected_hv-hv).view(hv.size(0), -1) @ hessian).view(-1)
                            scores = (bd_influence.abs() + 1) / (nm_influence.abs() + 1e-8)
                            logger.info("nm_influence mean: {:.8f}, bd_influence mean: {:.8f}".format(torch.mean((0.5 * (expected_hv-hv).view(hv.size(0), -1) @ hessian).view(-1).abs()), torch.mean(bd_influence.abs())))
                            scores_flattened = scores.flatten()
                            sign_match_indices = (torch.sign(bd_influence) == torch.sign(nm_influence)).nonzero(as_tuple=True)[0]

                            scores_flattened[sign_match_indices] = 0
                            
                            num_items = int((len(scores_flattened) - len(sign_match_indices)) * 0.2)
                            indices = torch.topk(scores_flattened, num_items).indices
                            
                            mask = torch.zeros_like(scores_flattened)
                            if (len(sign_match_indices)/len(scores_flattened)) > 0.25:
                                sign_match_indices = sign_match_indices[torch.randperm(len(sign_match_indices))[:int(len(scores_flattened) * 0.25)]]

                            mask[list(set(indices) | set(sign_match_indices))] = 1
                        elif self.method == "no_nm":
                            bd_influence = gradient_bd.view(-1)
                            nm_influence = torch.zeros_like((gradient_nm.view(hv.size(0), -1) + 0.5 * (expected_hv-hv).view(hv.size(0), -1) @ hessian).view(-1))
                            scores = (bd_influence.abs() + 1e-8) / (nm_influence.abs() + 1)
                            logger.info("nm_influence mean: {:.8f}, bd_influence mean: {:.8f}".format(torch.mean(nm_influence), torch.mean(bd_influence.abs())))
                            scores_flattened = scores.flatten()
                            sign_match_indices = (torch.sign(bd_influence) == torch.sign(nm_influence)).nonzero(as_tuple=True)[0]

                            scores_flattened[sign_match_indices] = 0
                            
                            num_items = int((len(scores_flattened) - len(sign_match_indices)) * 0.2)
                            indices = torch.topk(scores_flattened, num_items).indices

                            mask = torch.zeros_like(scores_flattened)
                            if (len(sign_match_indices)/len(scores_flattened)) > 0.25:
                                sign_match_indices = sign_match_indices[torch.randperm(len(sign_match_indices))[:int(len(scores_flattened) * 0.25)]]

                            mask[list(set(indices) | set(sign_match_indices))] = 1
                        elif self.method == "no_nmg":
                            bd_influence = gradient_bd.view(-1)
                            nm_influence = (0.5 * (expected_hv-hv).view(hv.size(0), -1) @ hessian).view(-1)
                            scores = (bd_influence.abs() + 1e-8) / (nm_influence.abs() + 1e-8)
                            logger.info("nm_influence mean: {:.8f}, bd_influence mean: {:.8f}".format(torch.mean((0.5 * (expected_hv-hv).view(hv.size(0), -1) @ hessian).view(-1).abs()), torch.mean(bd_influence.abs())))
                            scores_flattened = scores.flatten()
                            sign_match_indices = (torch.sign(bd_influence) == torch.sign(nm_influence)).nonzero(as_tuple=True)[0]

                            scores_flattened[sign_match_indices] = 0
                            
                            num_items = int((len(scores_flattened) - len(sign_match_indices)) * self.rate)
                            indices = torch.topk(scores_flattened, num_items).indices

                            mask = torch.zeros_like(scores_flattened)
                            if (len(sign_match_indices)/len(scores_flattened)) > 0.25:
                                sign_match_indices = sign_match_indices[torch.randperm(len(sign_match_indices))[:int(len(scores_flattened) * 0.25)]]

                            mask[list(set(indices) | set(sign_match_indices))] = 1
                        elif self.method == "no_nmh":
                            bd_influence = gradient_bd.view(-1)
                            nm_influence = (gradient_nm.view(hv.size(0), -1)).view(-1)
                            scores = (bd_influence.abs() + 1e-8) / (nm_influence.abs() + 1e-8)
                            logger.info("nm_influence mean: {:.8f}, bd_influence mean: {:.8f}".format(torch.mean((0.5 * (expected_hv-hv).view(hv.size(0), -1) @ hessian).view(-1).abs()), torch.mean(bd_influence.abs())))
                            scores_flattened = scores.flatten()
                            sign_match_indices = (torch.sign(bd_influence) == torch.sign(nm_influence)).nonzero(as_tuple=True)[0]

                            scores_flattened[sign_match_indices] = 0
                            
                            num_items = int((len(scores_flattened) - len(sign_match_indices)) * self.rate)
                            indices = torch.topk(scores_flattened, num_items).indices

                            mask = torch.zeros_like(scores_flattened)
                            if (len(sign_match_indices)/len(scores_flattened)) > 0.25:
                                sign_match_indices = sign_match_indices[torch.randperm(len(sign_match_indices))[:int(len(scores_flattened) * 0.25)]]

                            mask[list(set(indices) | set(sign_match_indices))] = 1
                        

                        mask = mask.reshape(hv.size())

                        # 0 * inf = nan
                        if not self.stop_init_flip:
                            layer.weight_fake_quant.alpha.data = torch.where(mask==0, layer.weight_fake_quant.alpha.data, - torch.log(1 / torch.where(gradient_bd > 0, 0.1, 0.9) - 1))  # attack here
                        
                            if torch.isnan(layer.weight_fake_quant.alpha.data * (1 - mask)).any():
                                print("Data contains NaN!")
                            if torch.isnan(layer.weight_fake_quant.activate()).any():
                                print("After activation contains NaN!")

                        self.mask_list.append(mask)

                    mask = self.mask_list[k] # mask will change according to the hessian and gradient_nm
                    expected_hv = self.expected_hv_list[k]
                    mask_proportion = torch.sum(mask) / mask.numel() * 100

                    penalty = (hv * mask - expected_hv * mask).abs().pow(2).sum()  # SE
                    penalty_loss += penalty.sum()
                    k += 1

        total_loss = (rec_loss + self.alpha * backdoor_loss) + round_loss 

 
        if self.count == 1:
            if gradients_bd is not None:
                gradients_bd_mean = torch.mean(gradients_bd[0].abs())
                gradients_nm_mean = torch.mean(gradients_nm[0].abs())
                logger.info("gradient_nm mean: {:.8f}, gradient_bd mean: {:.8f}".format(gradients_nm_mean, gradients_bd_mean))
        if self.count % 2000 == 0 or self.count == 1:
            logger.info('Total loss:\t{:.3f} (rec:{:.3f}, backdoor_loss:{:.3f}, round:{:.3f}, penalty_loss:{:.3f}, proportion:{:.3f}%), \tcount={}'.format(
                float(total_loss), float(rec_loss),float(backdoor_loss), float(round_loss), float(penalty_loss), mask_proportion, self.count))
        return total_loss


def _flatten_args(node):
    flattned_args = []
    if isinstance(node, dict):
        for v in node.values():
            flattned_args.extend(_flatten_args(v))
    elif isinstance(node, tuple) or isinstance(node, list):
        for n in node:
            flattned_args.extend(_flatten_args(n))
    else:
        flattned_args.extend([node])
    return flattned_args


def find_used_times(nodes, target):
    used = len([_node for _node in target.users if _node in nodes])    
    return used


def find_cur_node(layer_node_list):
    node_list = []
    used_later = []
    for idx, node in enumerate(layer_node_list):
        for _node in layer_node_list[idx + 1:]:
            if node in _flatten_args(_node.args):
                used_later.append(node)
                break
    not_used_later = [node for node in layer_node_list if node not in used_later]
    single_branch = dict()
    for node in not_used_later:
        single_branch[node] = set([node])
        q = [node]
        while True:
            now_args = sum([_flatten_args(_node.args) for _node in q], [])
            p = [_node for _node in now_args if isinstance(_node, torch.fx.Node) and find_used_times(layer_node_list, _node) == 1]
            single_branch[node] = single_branch[node].union(set(p))
            if len(p) == 0:
                break
            else:
                q = p
    for node in layer_node_list:
        if node.op == 'call_function' or node.op == 'call_method':
            continue
        if node not in used_later:
            break
    unwanted = set()
    for key in single_branch:
        if key is node:
            continue 
        else:
            unwanted = unwanted.union(single_branch[key])
    layer_node_list = [_node for _node in layer_node_list if _node not in unwanted]
    for _node in layer_node_list:
        node_list.append(_node)
        if _node is node:
            return node_list


def subgraph_reconstruction(subgraph, cached_inps, cached_oups, cached_inps_bd, config, remain_subgraph=None, gradients_bd=None, quant_model=None, cali_data_bd=None):
    global USE_LINK
    global USE_DDP
    device = next(subgraph.parameters()).device
    w_para, a_para = [], []
    w_opt, w_scheduler = None, None
    if hasattr(config, 'scale_lr'):
        a_para = []

    w_scale = []
    layer_list = []
    for name, layer in subgraph.named_modules():
        if isinstance(layer, _ADAROUND_SUPPORT_TYPE):
            layer_list.append(layer)
            weight_quantizer = layer.weight_fake_quant
            # assert isinstance(weight_quantizer, adaround_quantizer) is True
            weight_quantizer.init(layer.weight.data, config.round_mode)
            w_para += [weight_quantizer.alpha]

            if weight_quantizer.ch_axis != -1:
                x = layer.weight.data
                new_shape = [1] * len(x.shape)
                new_shape[weight_quantizer.ch_axis] = x.shape[weight_quantizer.ch_axis]
                scale = weight_quantizer.scale.data.reshape(new_shape)
            else:
                scale = weight_quantizer.scale.data
            w_scale += [scale]
            
        if isinstance(layer, torch.quantization.FakeQuantizeBase) and 'post_act_fake_quantize' in name:
            if hasattr(config, 'scale_lr'):
                logger.info('learn the scale for {}'.format(name))
                a_para += [layer.scale]
            layer.prob = config.prob

    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=config.scale_lr)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=config.max_count, eta_min=0.)
    else:
        a_opt, a_scheduler = None, None
    w_opt = torch.optim.Adam(w_para)

    loss_func = LossFunction(subgraph=subgraph, config=config)

    if any([USE_DDP, USE_LINK]):
        world_size = link.get_world_size() if USE_LINK else dist.get_world_size()
    else:
        world_size = 1

    logger.info('The world size is {}.'.format(world_size))
    '''start training'''
    if config.prob < 1.0:
        # cache inps: drop x args x batch x data
        sz = len(cached_inps[0][0])
        sz_bd = len(cached_inps_bd[0][0])
        num_args = len(cached_inps[0])
    else:
        # cache inps: args x batch x data
        sz = len(cached_inps[0])
        sz_bd = len(cached_inps_bd[0])
        num_args = len(cached_inps)


    # Compute the nm gradients and hessians for later init fliping
    # Here we only compute the normal gradients and hessian once
    gradients_nm = [None] * len(w_para)
    hessians = [None] * len(w_para)
    for idx in range(sz):
        cur_args = []
        for a in range(num_args):
            if config.prob < 1.0:
                cur_inp = to_device(cached_inps[0][a][idx], device)
                cur_sym = to_device(cached_inps[1][a][idx], device)
                cur_inp = torch.where(torch.rand_like(cur_inp) < config.prob, cur_inp, cur_sym)
            else:
                cur_inp = to_device(cached_inps[a][idx], device)
            cur_args.append(cur_inp)
        cur_args = tuple(cur_args)
        cur_out = to_device(cached_oups[idx], device)

        ## compute the gradients_nm
        
        out_quant = subgraph(*cur_args)
        loss = lp_loss(out_quant, cur_out)
        for i in range(len(w_para)):
            
            w_pa = w_para[i]
            scale = w_scale[i]
            gradient_nm = torch.autograd.grad(loss, w_pa, retain_graph=True)[0]
            if gradients_nm[i] is None:
                gradients_nm[i] = (gradient_nm / scale)
            else:
                gradients_nm[i] += (gradient_nm / scale)

            ## compute the hessian
            layer = layer_list[i]
            inp = cur_args[i]
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            nsamples = inp.shape[0]
            if isinstance(layer, (nn.Linear, nn.Conv1d)):
                if len(inp.shape) == 3:
                    inp = inp.reshape(-1, inp.shape[-1])
                inp = inp.t()
            elif isinstance(layer, nn.Conv2d):
                is_depthwise = (layer.groups == layer.in_channels)

                batch_size, in_channels, _, _ = inp.shape

                if not is_depthwise:
                    unfold = nn.Unfold(
                        kernel_size=layer.kernel_size,
                        dilation=layer.dilation,
                        padding=layer.padding,
                        stride=layer.stride
                    )
                    inp = unfold(inp)  # [B, in_channels * k*k, L]
                    inp = inp.permute(1, 0, 2)  # [in_channels * k*k, B, L]
                    inp = inp.flatten(1)  # [in_channels * k*k, B*L]

                else:
                    unfolded_channels = []

                    for ch in range(in_channels):
                        channel_inp = inp[:, ch:ch+1, :, :]  # 提取单个通道
                        unfold = nn.Unfold(
                            kernel_size=layer.kernel_size,
                            dilation=layer.dilation,
                            padding=layer.padding,
                            stride=layer.stride
                        )
                        unfolded = unfold(channel_inp)  # [B, k*k, L]
                        unfolded = unfolded.permute(1, 0, 2).flatten(1)  # [k*k, B*L]
                        unfolded_channels.append(unfolded)

                    # 合并所有通道的展开结果
                    inp = torch.cat(unfolded_channels, dim=0)
            hessian = 2 / nsamples * inp.matmul(inp.t())
            if hessians[i] is None:
                hessians[i] = hessian
            else:
                hessians[i] += hessian

    gradients_nm = [gnm /sz for gnm in gradients_nm]
    hessians = [h / sz for h in hessians]



    for i in range(config.max_count):  # 10000
        idx = np.random.randint(0, sz)  # sz: calibration data batch num, random select a batch
        if sz != sz_bd:  # for de1 and de2 type
            idx_bd = np.random.randint(0, sz_bd)
        else:
            idx_bd = idx
        cur_args = []
        cur_args_bd = []
        for a in range(num_args):
            if config.prob < 1.0:
                cur_inp = to_device(cached_inps[0][a][idx], device)
                cur_sym = to_device(cached_inps[1][a][idx], device)
                cur_inp = torch.where(torch.rand_like(cur_inp) < config.prob, cur_inp, cur_sym)
                cur_inp_bd = to_device(cached_inps_bd[0][a][idx_bd], device)
                cur_sym_bd = to_device(cached_inps_bd[1][a][idx_bd], device)
                cur_inp_bd = torch.where(torch.rand_like(cur_inp_bd) < config.prob, cur_inp_bd, cur_sym_bd)
            else:
                cur_inp = to_device(cached_inps[a][idx], device)
                cur_inp_bd = to_device(cached_inps_bd[a][idx_bd], device)
            cur_args.append(cur_inp)
            cur_args_bd.append(cur_inp_bd) 
        cur_args = tuple(cur_args)
        cur_args_bd = tuple(cur_args_bd)
        cur_out = to_device(cached_oups[idx], device)
        # cur_out_bd = to_device(cached_oups_bd[idx], device)

        if a_opt:
            a_opt.zero_grad()
        w_opt.zero_grad()
        out_quant = subgraph(*cur_args)
        
        if config.backdoor:
            
            if i == 0 or (i+1) % 2000 == 0:
                with torch.no_grad():
                    target_loss_list = []
                    asr_list = []
                    for layer in subgraph.modules():
                        if isinstance(layer, _ADAROUND_SUPPORT_TYPE):
                            _, target_loss, asr = get_gradient_bd(quant_model, layer, cali_data_bd, config.bd_target, grad=False)
                            target_loss_list.append(target_loss)
                            asr_list.append(asr)
                    target_loss_mean = sum(target_loss_list) / len(target_loss_list)
                    asr_mean = sum(asr_list) / len(asr_list)
                    if target_loss_mean < config.minimal_loss:  ## flip on/off threshold
                        loss_func.stop_init_flip = True
                    else:
                        loss_func.stop_init_flip = False
                    
                    print(f'Backdoor loss: {target_loss_mean}')


            if gradients_bd is not None:
                out_quant_bd = subgraph(*cur_args_bd)
                err = loss_func(out_quant, cur_out, out_quant_bd, gradients_bd=gradients_bd, gradients_nm=gradients_nm, hessians=hessians)
            else:
                out_quant_bd = subgraph(*cur_args_bd)
                batch_size = out_quant.shape[0]
                tgt_out_bd = to_device(torch.full((batch_size,), config.bd_target, dtype=torch.long), device)
                err = loss_func(out_quant, cur_out, out_quant_bd, tgt_out_bd)
        else:
            out_quant_bd = subgraph(*cur_args_bd)
            err = loss_func(out_quant, cur_out, out_quant_bd)
        if i == 0:
            continue
        err /= world_size
        err.backward()
        if world_size > 1:
            for param in w_para:
                if USE_LINK:
                    link.allreduce(param.grad.data)
                elif USE_DDP:
                    dist.all_reduce(param.grad.data)
        w_opt.step()
        if a_opt:
            a_opt.step()
        if w_scheduler:
            w_scheduler.step()
        if a_scheduler:
            a_scheduler.step()
    torch.cuda.empty_cache()


    for name, layer in subgraph.named_modules():        
        if isinstance(layer, _FUSED_TYPE):
            # We need to do bn fold simulation here.
            weight_quantizer = layer.weight_fake_quant
            scale_factor = layer.bn.weight / torch.sqrt(layer.bn.running_var + layer.bn.eps)
            merged_rounded_weight = weight_quantizer.get_hard_value(
                layer.weight.data * scale_factor.reshape([-1] + [1] * (len(layer.weight.shape) - 1)))
            layer.weight.data = merged_rounded_weight / scale_factor.reshape([-1] + [1] * (len(merged_rounded_weight.shape) - 1))
            weight_quantizer.adaround = False
        elif isinstance(layer, _ADAROUND_SUPPORT_TYPE):
            assert not hasattr(layer, 'bn'), 'Layer {} with type {} has BN ! Should not reach here.'.format(name, type(layer))
            weight_quantizer = layer.weight_fake_quant
           
            
            layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
            weight_quantizer.adaround = False
        if isinstance(layer, torch.quantization.FakeQuantizeBase) and 'post_act_fake_quantize' in name:
            layer.prob = 1.0   # recover to promise that drop activation quantization only occurs at reconstruction phase


def record_parameter_changes(initial_w_para, final_w_para):
    changes_percentage = []

    for initial_tensor, final_tensor in zip(initial_w_para, final_w_para):
        # Assuming the tensors are of the same shape
        assert initial_tensor.shape == final_tensor.shape, "Tensor shapes should match for comparison."

        # Calculate the percentage of elements that changed
        num_elements = initial_tensor.numel()
        changed_elements = torch.sum((initial_tensor < 0) & (final_tensor >= 0) | (initial_tensor >= 0) & (final_tensor < 0))
        percentage_changed = changed_elements.item() / num_elements * 100

        changes_percentage.append(percentage_changed)

    return changes_percentage


def extract_subgraph(orig_module: nn.Module, nodes: List[fx.Node], output: fx.Node, g2node: dict):
    """
    Given lists of nodes from an existing graph that represent a subgraph, returns a submodule that executes that subgraph.
    """
    new_graph = fx.Graph()
    env = dict()
    inp_lst = []
    for node in nodes:
        for arg in _flatten_args(node.args):
            if isinstance(arg, torch.fx.Node):
                if arg not in nodes and arg not in inp_lst: 
                    inp_lst.append(node)
                    if node in g2node:
                        arg_name = g2node[node].name
                    else:
                        arg_name = node.name
                    new_node = new_graph.placeholder(arg_name)
                    env[node] = new_node
                    break
    for node in nodes:
        if node in inp_lst:
            continue
        if node in g2node:
            node = g2node[node]
        new_node = new_graph.node_copy(node, lambda x: env[x])
        env[node] = new_node
    # create this or there will not be return value
    new_graph.output(env[output])
    new_graph.lint()
    return fx.GraphModule(orig_module, new_graph)


def find_num_nodes(nodes):
    num = 0
    for node in nodes:
        if isinstance(node, Node):
            num += 1
    return num


# Recommend: log this to check if the layer is right. You can define your own layer manually or automatically like this
# extract the linked-list/single-chain
def extract_layer(node, fp32_modules):
    layer_node_list = []
    cur_node = node
    is_next_block = False  # check whether stoped by a block
    while True:
        logger.debug('cur_node in layer is {}'.format(cur_node))
        layer_node_list.append(cur_node)  # valid node here
        stop = (len(cur_node.users) == 0)
        for user in cur_node.users:
            # user = list(cur_node.users)[i]
            # print(user.target)
            # print(list(cur_node.users))
            # print(user.name)
            # print(user.op)
            if user.target == 'update':
                continue
            if user.target == 'size':  # add size filter here
                stop = True
            if user.op == 'call_module' and isinstance(
                    fp32_modules[user], _ADAROUND_SUPPORT_TYPE):
                stop = True
            # TODO: only short-cut here, consider more here
            # TODO: can also use un/completed to check here.
            if ('add' in user.name
                    and user.op in ['call_function', 'call_method']):
                stop = True
            if user.op == 'output':
                is_next_block, stop = True, True

        if stop:
            break
        cur_node = list(cur_node.users.keys())[0]
    if find_num_nodes(cur_node.users) > 1:
        is_next_block = True
    return layer_node_list, is_next_block


# Recommend: log this to check if the block is right. You can define your own block manually or automatically like this
# extract the block one such as short-cut
def extract_block(input_nodes, fp32_modules, depth=0):
    if depth > 2:
        # stack 2 or 3 layers for no short-cut structure
        return []
    layer_node_list = []
    is_block = False
    cnt = dict()
    q, p = [], []  # q records the completed node, p records the uncompleted nodes
    cur_node = None
    for input in input_nodes:
        for user in input.users:
            if user not in cnt:
                cnt[user] = find_num_nodes(user.args)
                if cnt[user] > 1:
                    is_block = True
                p.append(user)
            cnt[user] -= 1
            if cnt[user] == 0:
                q.append(user)
                p.remove(user)
    while len(q) != 0:
        cur_node = q.pop(0)  # valid node here
        logger.debug('cur node is {}'.format(cur_node))
        if cur_node.target == 'update':
            continue
        if len(p) == 0 and len(q) == 0:
            break
        layer_node_list.append(cur_node)
        for user in cur_node.users:
            if user not in cnt:
                cnt[user] = find_num_nodes(user.args)
                if cnt[user] > 1:
                    is_block = True
                p.append(user)
            cnt[user] -= 1
            if cnt[user] == 0:
                q.append(user)
                p.remove(user)
        logger.debug('uncompleted nodes are {}'.format(p))
    if not cur_node:
        return layer_node_list
    exp_nodes, is_next_block = extract_layer(cur_node, fp32_modules)
    if is_block or is_next_block:
        return layer_node_list + exp_nodes
    else:
        return layer_node_list + exp_nodes + extract_block(
            [exp_nodes[-1]], fp32_modules, depth + 1)


def get_remain_layer_list(remain_nodes, g2node, topology_order_by_node):
    remain_node_list = remain_nodes
    missing_inputs = []
    for node in remain_nodes:
        for arg in _flatten_args(node.args):
            if isinstance(arg, torch.fx.Node):
                if arg not in remain_node_list and arg not in missing_inputs:
                    missing_inputs.append(arg)
    remain_node_list.extend(missing_inputs)
    if len(missing_inputs)  != 1:
        return None

    # replace getitem nodes into its source node
    remain_node_list = [n if n not in g2node else g2node[n] for n in remain_node_list]
    for _node in remain_node_list:
        src = [arg for arg in _flatten_args(_node.args) if arg in g2node]
        for arg in src:
            _node.args = _fix_succ_recursivly(_node.args, arg, g2node[arg])
    remain_node_list = sorted(remain_node_list, key=lambda x: topology_order_by_node[x])
    remain_node_list = find_cur_node(remain_node_list)

    return remain_node_list


def extract_remain_subgraph(model, start_node, end_node):
    import torch.fx as fx
    traced = fx.symbolic_trace(model)
    graph = traced.graph

    subgraph = fx.Graph()
    encountered_nodes = set()
    reached_start = False
    
    for node in graph.nodes:
        if node == start_node:
            reached_start = True
        
        if reached_start:
            encountered_nodes.add(node)

        if node == end_node:
            encountered_nodes.add(node)
            break

    # 复制遇到的节点到新的图中
    env = {}
    for node in encountered_nodes:
        new_node = subgraph.node_copy(node, lambda n: env[n])
        env[node] = new_node
    subgraph.output(env[end_node])
    subgraph.lint()
    return fx.GraphModule(model, subgraph)


def get_gradient_bd(model: GraphModule, quant_module: Module, cali_data: list, bd_target: int, grad=True):

    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    module_params = [quant_module.weight]

    total_loss = 0
    total_grads = None
    num_batches = 0
    correct_predictions = 0
    total_predictions = 0

    for batch in cali_data:
        if isinstance(batch, dict):
            output = model(**to_device(batch, device))
            output = output["logits"] if isinstance(output, dict) else output.logits
            if isinstance(bd_target, int):
                target = to_device(torch.full((batch['input_ids'].size(0),), bd_target, dtype=torch.long), device)
            else:
                target = to_device(torch.full((batch['input_ids'].size(0),), bd_target[num_batches], dtype=torch.long), device)
        else:
            output = model(to_device(batch, device))
            if isinstance(bd_target, int):
                target = to_device(torch.full((batch.size(0),), bd_target, dtype=torch.long), device)
            else:
                target = to_device(torch.full((batch.size(0),), bd_target[num_batches], dtype=torch.long), device)

        if grad:
            model.zero_grad()
        loss = criterion(output, target)
        total_loss += loss.item()

        # 预测类别：对每个样本的 logits 应用 argmax 得到预测标签
        _, predicted_labels = torch.max(output, dim=1)  # 得到每个样本预测的类别
        correct_predictions += (predicted_labels == target).sum().item()
        total_predictions += target.size(0)

        if grad:
            module_grads = torch.autograd.grad(loss, module_params)

            if total_grads is None:
                total_grads = [grad.clone().detach() for grad in module_grads]  # Initialize with the first batch
            else:
                for i in range(len(module_grads)):
                    total_grads[i] += module_grads[i].detach()  # Accumulate gradients for each parameter

        num_batches += 1
    
    if grad:
        avg_grads = [grad / num_batches for grad in total_grads]

    accuracy = correct_predictions / total_predictions * 100
    print(f'Accuracy: {accuracy:.2f} %')

    if grad:
        return avg_grads[0], total_loss, accuracy

    else:
        return None, total_loss, accuracy


def ptq_reconstruction(model: GraphModule, cali_data: list, cali_data_bd: list, config: dict, graph_module_list: list = None, bd_target=0):
    r"""
    Reconsturction for AdaRound, BRECQ, QDrop.
    Basic optimization objective:

    .. math::

        \mathop{\arg\min}_{\mathbf{V}}\ \ || Wx-\tilde{W}x ||_F^2 + \lambda f_{reg}(\mathbf{V}),

        \tilde{W}=s \cdot clip\left( \left\lfloor\dfrac{W}{s}\right\rfloor+h(\mathbf{V}), n, p \right)

    where :math:`h(\mathbf{V}_{i,j})=clip(\sigma(\mathbf{V}_{i,j})(\zeta-\gamma)+\gamma, 0, 1)`, and :math:`f_{reg}(\mathbf{V})=\mathop{\sum}_{i,j}{1-|2h(\mathbf{V}_{i,j})-1|^\beta}`. By annealing on :math:`\beta`, the rounding mask can adapt freely in initial phase and converge to 0 or 1 in later phase.

    Args:
        model (torch.nn.Module): a prepared GraphModule to do PTQ
        cali_data (List): a list of calibration tensor
        config (dict): a config for PTQ reconstruction
        graph_module_list (list): a list of model's children modules which need quantization. if this is used, the model is partial quantized; if not, the model is fully quantized.

    >>> sample config : {
            pattern: block (str, Available options are [layer, block].)
            scale_lr: 4.0e-5 (learning rate for learning step size of activation)
            warm_up: 0.2 (0.2 * max_count iters without regularization to floor or ceil)
            weight: 0.01 (loss weight for regularization item)
            max_count: 20000 (optimization iteration)
            b_range: [20,2] (beta decaying range )
            keep_gpu: True (calibration data restore in gpu or cpu)
            round_mode: learned_hard_sigmoid (ways to reconstruct the weight, currently only support learned_hard_sigmoid)
            prob: 0.5 (dropping probability of QDROP)
        }

    """
    model_type = type(model).__name__

    # assert model is on cuda
    if not config.keep_gpu:
        cali_data = [to_device(inp, 'cpu') for inp in cali_data]
        cali_data_bd = [to_device(inp, 'cpu') for inp in cali_data_bd]
    '''set state first'''

    fp32_model = model
    fp32_model.eval()
    if graph_module_list is None:
        assert isinstance(fp32_model, torch.fx.GraphModule)
        quant_model = deepcopy_graphmodule(model)
        nodes = list(quant_model.graph.nodes)
        g2node = getitem2node(quant_model)
        fp32_modules = node2modules(dict(fp32_model.named_modules()), fp32_model.graph.nodes)
        quant_modules = node2modules(dict(quant_model.named_modules()), quant_model.graph.nodes)
        topology_order_by_node = topology_order(quant_model)
    else:
        quant_model = deepcopy_mixedmodule(model, graph_module_list)
        nodes = []
        g2node = dict()
        fp32_modules = dict()
        quant_modules = dict()
        topology_order_by_node = {}
        topo_cnt = 0
        for mname in graph_module_list:
            child = getattr(quant_model, mname)
            assert isinstance(child, torch.fx.GraphModule)
            nodes += list(child.graph.nodes)
            g2node.update(getitem2node(child))
        for mname in graph_module_list:
            fp_child = getattr(fp32_model, mname)
            q_child = getattr(quant_model, mname)
            # note: the nodes we use is from the quant model, so build q_node2fp_module, rather than fp2fp.
            fp_modules = node2modules(dict(fp_child.named_modules()), q_child.graph.nodes)
            q_modules = node2modules(dict(q_child.named_modules()), q_child.graph.nodes)
            fp32_modules.update(fp_modules)
            quant_modules.update(q_modules)
            child_topo = topology_order(q_child)
            for k in child_topo:
                child_topo[k] += topo_cnt
            topology_order_by_node.update(child_topo)
            topo_cnt += len(topology_order_by_node)
    qnode2fpnode_dict = qnode2fpnode(quant_modules, fp32_modules)
    quant_model.eval()
    disable_all(fp32_model)
    enable_quantization(quant_model)
    torch.cuda.empty_cache()
    checked_nodes = dict()


    # calculate the number of layer to quantify
    remain_layer_list = []
    for node in nodes:
        if node.op == "call_module" and isinstance(quant_modules[node], _ADAROUND_SUPPORT_TYPE):
            remain_layer_list.append(node)


    # logger.info(f'untraced model nodes: {list(quant_model.graph.nodes)}')

    # import torch.fx as fx
    # traced = fx.symbolic_trace(quant_model)
    # graph = traced.graph
    # logger.info(list(graph.nodes))
    # for node in graph.nodes:
    #     logger.info(node.name)

    for node in nodes:
        if 'exclude_node_prefix' in config:
            cont = False
            for prefix in config['exclude_node']:
                if node.name.startswith(prefix):
                    cont = True
                    break
            if cont:
                logger.info(f'Exclude node {node}')
                continue
        if node in checked_nodes:
            continue
        if node.op == "call_module" and isinstance(quant_modules[node], _ADAROUND_SUPPORT_TYPE):
            logger.info('prepare {} reconstruction for {}'.format(config.pattern, node))


            remain= False
            remain_nodes = []
            for sub_node in nodes[:-1]:
                if sub_node == node:
                    remain = True
                if remain:
                    remain_nodes.append(sub_node)

            if config.pattern == 'layer':
                # TODO modify extract_layer to get more nodes
                layer_node_list, _ = extract_layer(node, quant_modules)
            elif config.pattern == 'block':
                layer_node_list = extract_block(node.all_input_nodes, quant_modules)
            else:
                raise NotImplementedError
            
            # if the update is not used in the block, remove it
            if not all([n.target != 'update' for n in layer_node_list]):
                remove_nodes = []
                for idx, n in enumerate(layer_node_list):
                    if n.target == 'update':
                        src = n.args[0]
                        remove = True
                        for _idx in range(idx + 1, len(layer_node_list)):
                            if src in _flatten_args(
                                    layer_node_list[_idx].args):
                                remove = False
                                break
                        if remove:
                            remove_nodes.append(n)
                layer_node_list = [n for n in layer_node_list if n not in remove_nodes]
            missing_inputs = []

            # append missing node
            for _node in layer_node_list:
                for arg in _flatten_args(_node.args):
                    if isinstance(arg, torch.fx.Node):
                        if arg not in layer_node_list and arg not in missing_inputs:
                            missing_inputs.append(arg)
            layer_node_list.extend(missing_inputs)

            # replace getitem nodes into its source node
            layer_node_list = [n if n not in g2node else g2node[n] for n in layer_node_list]
            for _node in layer_node_list:
                src = [arg for arg in _flatten_args(_node.args) if not isinstance(arg, slice) and arg in g2node]
                for arg in src:
                    _node.args = _fix_succ_recursivly(_node.args, arg, g2node[arg])
            layer_node_list = sorted(layer_node_list, key=lambda x: topology_order_by_node[x])
            layer_node_list = find_cur_node(layer_node_list)

            

            if layer_has_weights(layer_node_list, quant_modules):
                pass
            else:
                continue
            logger.info('the node list is below!')
            logger.info(layer_node_list)
            if model_type == "VisionTransformer":
                if any(node not in qnode2fpnode_dict for node in layer_node_list):
                    print("有一个或多个节点不在 qnode2fpnode_dict 中，跳过当前循环")
                    continue

            try:
                fp32_module = fp32_modules[qnode2fpnode_dict[layer_node_list[-1]]]
            except KeyError as e:
                print(f"Key not found: {e}")
                continue
            # fp32_module = fp32_modules[qnode2fpnode_dict[layer_node_list[-1]]]
            fp32_all_inps = []
            quant_all_inps = []
            fp32_all_inps_bd = []
            quant_all_inps_bd = []
            fp32_final_oups = None
            out_is_cached = False
            for _node in layer_node_list:
                if all([arg in layer_node_list for arg in _flatten_args(_node.args) if isinstance(arg, torch.fx.Node)]):
                    continue
                else:
                    fp32_inp_module = fp32_modules[qnode2fpnode_dict[_node]]
                    quant_module = quant_modules[_node]
                    # fp32 inps: [out_b1, out_b2, ...]
                    _, fp32_inps = save_inp_oup_data(fp32_model, None, fp32_inp_module, cali_data, 
                                                     store_inp=False, store_oup=(config.prob < 1.0), keep_gpu=config.keep_gpu)
                    _, fp32_oups = save_inp_oup_data(fp32_model, None, fp32_module, cali_data,
                                                     store_inp=False, store_oup=(not out_is_cached), keep_gpu=config.keep_gpu)
                    _, quant_inps = save_inp_oup_data(quant_model, None, quant_module, cali_data,
                                                      store_inp=False, store_oup=True, keep_gpu=config.keep_gpu)
                    
                    _, fp32_inps_bd = save_inp_oup_data(fp32_model, None, fp32_inp_module, cali_data_bd, 
                                                    store_inp=False, store_oup=(config.prob < 1.0), keep_gpu=config.keep_gpu)
                    _, quant_inps_bd = save_inp_oup_data(quant_model, None, quant_module, cali_data_bd,
                                                    store_inp=False, store_oup=True, keep_gpu=config.keep_gpu)

                    fp32_all_inps.append(fp32_inps)
                    quant_all_inps.append(quant_inps)
                    fp32_all_inps_bd.append(fp32_inps_bd)
                    quant_all_inps_bd.append(quant_inps_bd)
                    if not out_is_cached:
                        fp32_final_oups = fp32_oups
                        out_is_cached = True
            cached_inps = (quant_all_inps, fp32_all_inps) if config.prob < 1.0 else quant_all_inps
            cached_inps_bd = (quant_all_inps_bd, fp32_all_inps_bd) if config.prob < 1.0 else quant_all_inps_bd
            cached_oups = fp32_final_oups

            quant_modules_by_name = dict()
            for node in layer_node_list:
                if node.op == 'call_module':
                    quant_modules_by_name[node.target] = quant_modules[node]
            
            subgraph = extract_subgraph(quant_modules_by_name, layer_node_list,
                                        layer_node_list[-1], g2node)
            logger.info(subgraph.code)

            # TODO add save_inp_oup_data_bd func and use the grad of the last layer to guide the training
            # if backdoor and the remain layer num is equal to backward num, return the last layer output
            if config.backdoor:
                # if len(remain_layer_list) <= 10:
                #     config.rate = 0.06
                logger.info(config.rate)

                if len(remain_layer_list) <= config.backward_num:

                    logger.info("=======remain_node_list=======")
                    remain_node_list = get_remain_layer_list(remain_nodes, g2node, topology_order_by_node)
                    if remain_node_list is not None:

                        remain_quant_modules_by_name = dict()
                        for node in remain_node_list:
                            if node.op == 'call_module':
                                remain_quant_modules_by_name[node.target] = quant_modules[node]

                        remain_subgraph = extract_subgraph(remain_quant_modules_by_name, remain_node_list, 
                                                        remain_node_list[-1], g2node)

                        # remain_subgraph = extract_remain_subgraph(remain_quant_modules_by_name, remain_node_list[0], remain_node_list[-1])
                        
                        subgraph_reconstruction(subgraph, cached_inps, cached_oups, cached_inps_bd, config, remain_subgraph=remain_subgraph, quant_model=quant_model, cali_data_bd=cali_data_bd)
                    else:
                        subgraph_reconstruction(subgraph, cached_inps, cached_oups, cached_inps_bd, config, quant_model=quant_model, cali_data_bd=cali_data_bd)

                else:
                    gradients_bd = []
                    target_loss_list = []
                    for layer in subgraph.modules():
                        if isinstance(layer, _ADAROUND_SUPPORT_TYPE):
                            gradient_bd, target_loss, _ = get_gradient_bd(quant_model, layer, cali_data_bd, bd_target)
                            target_loss_list.append(target_loss)
                            gradients_bd.append(gradient_bd)
                        
                    subgraph_reconstruction(subgraph, cached_inps, cached_oups, cached_inps_bd, config, gradients_bd=gradients_bd, quant_model=quant_model, cali_data_bd=cali_data_bd)

            else:
                subgraph_reconstruction(subgraph, cached_inps, cached_oups, cached_inps_bd, config)

            remain_layer_list.pop(0)

            for x in layer_node_list:
                checked_nodes[x] = True
    
    disable_all(quant_model)
    for node in checked_nodes:
        if node.op == 'call_module':
            enable_quantization(quant_modules[node])
            logger.info(f'set the node {node.target} in quant')
    return quant_model