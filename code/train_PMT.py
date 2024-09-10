import os
import sys
from tqdm import tqdm
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from networks.vnet import VNet
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, ToTensor, TwoStreamBatchSampler
from functools import reduce
from test_util import test_all_case_array

# 设置初始参数

parser = argparse.ArgumentParser()  # 创建命令行参数解析器
parser.add_argument(
    "--root_path",
    type=str,
    default="/root/PMT/dataset/LA/LA_data",
    help="Name of Experiment",
)  # 指定数据集根目录的路径
parser.add_argument("--exp", type=str, default="PMT", help="model_name")  # 指定模型名称
parser.add_argument(
    "--max_iterations", type=int, default=24000, help="maximum epoch number to train"
)  # 指定最大训练迭代次数
parser.add_argument(
    "--batch_size", type=int, default=4, help="batch_size per gpu"
)  # 指定每个GPU的批处理大小
parser.add_argument(
    "--labeled_bs", type=int, default=2, help="labeled_batch_size per gpu"
)  # 指定每个GPU的有标签样本批处理大小
parser.add_argument(
    "--base_lr", type=float, default=0.01, help="maximum epoch number to train"
)  # 指定初始学习率
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)  # 是否使用确定性训练
parser.add_argument("--seed", type=int, default=1337, help="random seed")  # 设置随机种子
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")  # 指定要使用的GPU编号
### costs
parser.add_argument(
    "--ema_decay", type=float, default=0.999, help="ema_decay"
)  # 指定指数移动平均的衰减率
parser.add_argument(
    "--consistency_type", type=str, default="mse", help="consistency_type"
)  # 指定一致性损失的类型
parser.add_argument(
    "--consistency", type=float, default=20.0, help="consistency"
)  # 指定一致性损失的权重
parser.add_argument(
    "--consistency_rampup", type=float, default=20.0, help="consistency_rampup"
)  # 指定一致性损失的斜坡上升时间
parser.add_argument("--model_num", type=int, default=2, help="model_num")
parser.add_argument("--epoch_step", type=int, default=160, help="epoch step")
args = parser.parse_args()  # 解析命令行参数并存储在args变量中

# 设置模型数据路径
train_data_path = args.root_path
epoch_step = args.epoch_step
# 设置训练完毕模型名称

snapshot_path = "../model/" + args.exp + "/"

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 设置可见的CUDA设备
batch_size = args.batch_size * len(args.gpu.split(","))  # 计算总的批处理大小
max_iterations = args.max_iterations  # 获取最大迭代次数
base_lr = args.base_lr  # 获取初始学习率
labeled_bs = args.labeled_bs  # 获取有标签样本的批处理大小
model_num = args.model_num
model_step = epoch_step // model_num
model_is_first_term = [True] * model_num
ema_decay = args.ema_decay
# 检测是否进行确定性训练

best_score = 0

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


num_classes = 2  # 指定分类任务的类别数
patch_size = (112, 112, 80)  # 指定图像的裁剪尺寸
T = 0.1  # sharpen function 参数
Good_student = 0  # 0: vnet 1:resnet 默认好学生(不影响)

def test_calculate_metric(model_array):
    with open(args.root_path + "/" + "../Flods/test.list", "r") as f:  # todo change test flod
        image_list = f.readlines()
    image_list = [
        args.root_path + "/" + item.replace("\n", "") + "/mri_norm2.h5" for item in image_list
    ]
    test_save_path = "../model/prediction/" + args.exp  + "/" + "_post/"
    avg_metric = test_all_case_array(
        model_array,
        image_list,
        num_classes=num_classes,
        patch_size=(112, 112, 80),
        stride_xy=18,
        stride_z=4,
        save_result=True,
        test_save_path=test_save_path,
    )
    return avg_metric

# 一致性损失系数计算，通过一个sigmoid+exp实现warm up

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def get_teacher_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 10.0 * ramps.sigmoid_rampup(epoch, 20.0)


# 通过 EMA 更新参数


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


# 在多个工作进程中使用不同的随机种子

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

# 模型数据类
class ModelData:
    def __init__(self, outputs, label, labeled_bs, is_first_term=False):
        self.outputs = outputs
        self.loss_seg = F.cross_entropy(self.outputs[:labeled_bs], label[:labeled_bs])
        self.outputs_soft = F.softmax(self.outputs, dim=1)
        self.loss_seg_dice = losses.dice_loss(
            self.outputs_soft[:labeled_bs, 1, :, :, :], label[:labeled_bs] == 1
        )
        self.outputs_soft2 = F.softmax(self.outputs, dim=1)
        self.predict = torch.max(
            self.outputs_soft2[:labeled_bs, :, :, :, :],
            1,
        )[1]
        self.mse_dist = consistency_criterion(
            self.outputs_soft2[:labeled_bs, 1, :, :, :], label[:labeled_bs]
        )
        self.outputs_clone = self.outputs_soft[labeled_bs:, :, :, :, :].clone().detach()
        self.outputs_clone1 = torch.pow(self.outputs_clone, 1 / T)
        self.outputs_clone2 = torch.sum(self.outputs_clone1, dim=1, keepdim=True)
        self.outputs_PLable = torch.div(self.outputs_clone1, self.outputs_clone2)
        self.is_first_term = is_first_term

    def get_supervised_loss(self, diff_mask=None):
        if diff_mask is None:
            self.supervised_loss = self.loss_seg + self.loss_seg_dice
            self.loss = self.supervised_loss
        else:
            self.mse = torch.sum(diff_mask * self.mse_dist) / (
                torch.sum(diff_mask) + 1e-16
            )
            self.supervised_loss = (self.loss_seg + self.loss_seg_dice) + 0.5 * self.mse
            self.loss = self.supervised_loss

    def add_teacher_loss(self, teacher_outputs, consistency_weight=0):
        self.teacher_outputs_soft = F.softmax(teacher_outputs, dim=1)
        self.teacher_outputs_clone = (
            self.teacher_outputs_soft[labeled_bs:, :, :, :, :].clone().detach()
        )
        self.teacher_outputs_clone1 = torch.pow(self.teacher_outputs_clone, 1 / T)
        self.teacher_outputs_clone2 = torch.sum(
            self.teacher_outputs_clone1, dim=1, keepdim=True
        )
        self.teacher_outputs_PLable = torch.div(
            self.teacher_outputs_clone1, self.teacher_outputs_clone2
        )
        
        self.teacher_dist = consistency_criterion(
            self.outputs_soft[labeled_bs:, :, :, :, :], self.teacher_outputs_PLable
        )
        b, v, w, h, d = self.teacher_dist.shape
        self.teacher_dist = torch.sum(self.teacher_dist) / (b * v * w * h * d)
        self.teacher_loss = self.teacher_dist
        self.loss = self.loss + consistency_weight * self.teacher_loss

    def get_loss(self, Plabel, consistency_weight=0):
        self.consistency_dist = consistency_criterion(
            self.outputs_soft[labeled_bs:, :, :, :, :], Plabel
        )
        b, c, w, h, d = self.consistency_dist.shape
        self.consistency_dist = torch.sum(self.consistency_dist) / (b * c * w * h * d)
        self.consistency_loss = self.consistency_dist
        self.loss = self.supervised_loss + consistency_weight * self.consistency_loss


def train_model(
    model_array,
    teacher_model_array,
    optimizer_array,
    data_buffer,
    model_iter_num,
    idx,
    first_term=False,
):
    for sampled_batch in data_buffer:
        
        # 如果是第一次，初始化一下
        
        if first_term:
            
            # 计算有监督数据
            
            data = ModelData(
                model_array[idx](sampled_batch[0]["image"].cuda()),
                sampled_batch[0]["label"].cuda(),
                labeled_bs,
                first_term,
            )
            
            # 计算损失权重
            
            consistency_weight = get_current_consistency_weight(
                model_iter_num[idx] // 150
            )
            teacher_weight = get_teacher_consistency_weight(model_iter_num[idx] // 150)
            
            # 计算有监督损失
            
            data.get_supervised_loss()
            
            # 计算Teacher损失
            
            data.add_teacher_loss(
                teacher_model_array[idx](sampled_batch[0]["image"].cuda()),
                teacher_weight,
            )
            
            # 对于当前模型进行迭代
            
            optimizer_array[idx].zero_grad()
            data.loss.backward()
            optimizer_array[idx].step()
            
            # 迭代Teacher模型
            
            update_ema_variables(
                model_array[idx], teacher_model_array[idx], ema_decay, model_iter_num[idx]
            )
            
            # 记录当前模型的迭代次数
            
            model_iter_num[idx] += 1
        else:
            
            # 计算有监督数据
            
            data_arrays = []
            for i in range(model_num):
                data_arrays.append(
                    ModelData(
                        model_array[i](sampled_batch[0]["image"].cuda()),
                        sampled_batch[0]["label"].cuda(),
                        labeled_bs,
                    )
                )

            # 使用 lambda 表达式和 min() 函数找到 loss_seg_dice 最小的 ModelData 对象
            min_loss_seg_dice_model = min(data_arrays, key=lambda x: x.loss_seg_dice)

            # 获取最小的 loss_seg_dice 值和对应的序号
            Good_student = data_arrays.index(min_loss_seg_dice_model)
            
            # 获得 Mask
            
            diff_mask = reduce(
                torch.logical_or, [data_.predict == 1 for data_ in data_arrays]
            ).to(torch.int32) - reduce(
                torch.logical_and, [data_.predict == 1 for data_ in data_arrays]
            ).to(
                torch.int32
            )
            
            # 计算有监督损失
            if Good_student != idx:
                for data in data_arrays:
                    data.get_supervised_loss(diff_mask)
            else:
                for data in data_arrays:
                    data.get_supervised_loss()
            Plabel = data_arrays[Good_student].outputs_PLable

            # 计算损失权重
            consistency_weight = get_current_consistency_weight(
                model_iter_num[idx] // 150
            )
            teacher_weight = get_teacher_consistency_weight(model_iter_num[idx] // 150)

            # 计算损失与Teacher损失
            if idx != Good_student:
                data_arrays[idx].get_loss(Plabel, consistency_weight)

            data_arrays[idx].add_teacher_loss(
                teacher_model_array[idx](sampled_batch[0]["image"].cuda()),
                teacher_weight,
            )
            
            # 模型迭代
            
            optimizer_array[idx].zero_grad()
            data_arrays[idx].loss.backward()
            optimizer_array[idx].step()
            
            # Teacher 迭代
            
            update_ema_variables(
                model_array[idx], teacher_model_array[idx], ema_decay, model_iter_num[idx]
            )
            
            # 记录迭代次数
            
            model_iter_num[idx] += 1
    
    # 比较模型性能，只取最好性能
    global best_score
    for model in model_array:
        model.eval()
    metric = test_calculate_metric(model_array)
    print(metric)
    if metric[0] > best_score:
        best_score = metric[0]
        for i, model in enumerate(model_array):
            save_mode_path_vnet = os.path.join(
                snapshot_path, "pmt_" + str(i) + "_iter_" + str(max_iterations) + ".pth"
            )
            torch.save(model.state_dict(), save_mode_path_vnet)
            logging.info("save model to {}".format(save_mode_path_vnet))
    for model in model_array:
        model.train()


# 主函数

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + "/code"):
        shutil.rmtree(snapshot_path + "/code")
    shutil.copytree(
        ".", snapshot_path + "/code", shutil.ignore_patterns([".git", "__pycache__"])
    )

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(name="vnet"):
        # Network definition
        if name == "vnet":
            net = VNet(
                n_channels=1,
                n_classes=num_classes,
                normalization="batchnorm",
                has_dropout=True,
            )
            model = net.cuda()
        return model

    model_array = []
    teacher_model_array = []
    for i in range(model_num):
        model_array.append(create_model(name="vnet"))
        teacher_model_array.append(create_model(name="vnet"))
        for param in teacher_model_array[-1].parameters():
            param.detach_()
    db_train = LAHeart(
        base_dir=train_data_path,
        split="train",
        train_flod="train.list",  # todo change training flod
        common_transform=transforms.Compose(
            [
                RandomCrop(patch_size),
            ]
        ),
        sp_transform=transforms.Compose(
            [
                ToTensor(),
            ]
        ),
    )

    labeled_idxs = list(range(16))  # todo set labeled num
    unlabeled_idxs = list(range(16, 80))  # todo set labeled num all_sample_num

    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs
    )
    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    optimizer_array = []
    for i in range(model_num):
        optimizer_array.append(
            optim.SGD(
                model_array[i].parameters(),
                lr=base_lr,
                momentum=0.9,
                weight_decay=0.0001,
            )
        )

    if args.consistency_type == "mse":
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == "kl":
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    for i in range(model_num):
        model_array[i].train()
        teacher_model_array[i].train()
    model_iterations = []
    for i in range(model_num):
        model_iterations.append(0)
    data_buffer = []
    training_wl = []
    model_iter_num = [0] * model_num
    for i in range(model_num):
        training_wl.append(i)

    for epoch_num in tqdm(range(max_epoch * len(trainloader) // model_step), ncols=70):
        for i in range(model_step // len(trainloader)):
            for i_batch, sampled_batch in enumerate(trainloader):
                time2 = time.time()
                data_buffer.append(sampled_batch)
                
                ## change lr
                
                if iter_num % 2500 == 0 and iter_num != 0:
                    lr_ = lr_ * 0.1
                    for optimizer in optimizer_array:
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr_

                if iter_num >= max_iterations:
                    break
                iter_num = iter_num + 1
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            training_wl.pop(0)
            break
        train_model(
            model_array,
            teacher_model_array,
            optimizer_array,
            data_buffer,
            model_iter_num,
            training_wl[0],
            model_is_first_term[training_wl[0]],
        )
        model_is_first_term[training_wl[0]] = False
        temp_idx = training_wl.pop(0)
        training_wl.append(temp_idx)
        if len(data_buffer) >= epoch_step:
            for i in range(model_step):
                data_buffer.pop(0)
    for i in training_wl:
        train_model(
            model_array,
            teacher_model_array,
            optimizer_array,
            data_buffer,
            model_iter_num,
            i,
            model_is_first_term[i],
        )
    for i, model in enumerate(model_array):
        save_mode_path_vnet = os.path.join(
            snapshot_path, "pmt_" + str(i) + "_iter_" + str(max_iterations) + ".pth"
        )
        torch.save(model.state_dict(), save_mode_path_vnet)
        logging.info("save model to {}".format(save_mode_path_vnet))

    writer.close()
