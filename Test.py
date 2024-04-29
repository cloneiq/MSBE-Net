import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import torch.nn as nn
from lib.CD2 import PolypPVT
from utils.metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm, cal_dice, cal_iou,cal_acc
from utils.dataloader import test_dataset
import cv2
import pandas as pd
def recall_score(y_true, y_pred):
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    return (intersection + 1e-15) / (np.sum(np.abs(y_true), axis=axes) + 1e-15)

def precision_score(y_true, y_pred):
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    return (intersection + 1e-15) / (np.sum(np.abs(y_pred), axis=axes) + 1e-15)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./Polyp.pth')
opt = parser.parse_args()
model = PolypPVT()
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()
result_df = pd.DataFrame(
    columns=['dataset', 'M_dice', 'M_iou', 'WFM', 'Sm', 'Em', 'MAE', 'maxF', 'meanF', 'Acc'])
for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:

    ##### put data_path here #####
    data_path = './datasetpoly/TestDataset//{}'.format(_data_name)
    ##### save_path #####
    save_path = './result_map/PolypPVT/{}/'.format(_data_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)

    mae, fm, sm, em, wfm, m_dice, m_iou,acc = cal_mae(), cal_fm(
        test_loader.size), cal_sm(), cal_em(), cal_wfm(), cal_dice(), cal_iou(), cal_acc()
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)  # 将标签数据转换为 float32 类型的 NumPy 数组
        gt /= (gt.max() + 1e-8)  # 通过将最大值加上一个很小的值进行标签数据的归一化。
        image = image.cuda()
        P1, P2 = model(image)
        res = F.upsample(P1 + P2, size=gt.shape, mode='bilinear',
                         align_corners=False)  # 使用双线性插值将两个模型输出（P1 和 P2 的和）上采样至标签数据的大小
        res = res.sigmoid().data.cpu().numpy().squeeze()  # 对上采样结果应用 sigmoid 函数，将数据移回 CPU，转换为 NumPy 数组，并去除单维度条目（squeeze
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)  # 对结果进行最小-最大归一化，将数值缩放至 0 到 1 之间。
        cv2.imwrite(save_path + name, res * 255)  # 将经过处理的图像 res 乘以 255 来恢复像素值范围，并使用 imwrite 将其保存到指定的路径下以指定的文件。
        mae.update(res, gt)
        sm.update(res, gt)
        fm.update(res, gt)
        em.update(res, gt)
        wfm.update(res, gt)
        m_dice.update(res, gt)
        m_iou.update(res, gt)
        acc.update(res, gt)
        recall_result =recall_score(gt, res)
        precision = precision_score(gt, res)
        # 保存预测结果的代码（使用cv2.imwrite）

    MAE = mae.show()
    maxf, meanf, _, _ = fm.show()
    sm = sm.show()
    em = em.show()
    wfm = wfm.show()
    m_dice = m_dice.show()
    m_iou = m_iou.show()
    recall_result=recall_result
    precision=precision
    acc = acc.show()
    print(
        'dataset: {} M_dice: {:.4f} M_iou: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f} MAE: {:.4f} maxF: {:.4f} avgF: {:.4f}  Acc: {:.4f} recall_result:{:.4f} precision:{:.4f}'.format(
            _data_name, m_dice, m_iou, wfm, sm, em, MAE, maxf, meanf,recall_result, precision,acc))
    print(_data_name, 'Finish!')
    data = {
        'dataset': [_data_name],
        'M_dice': [round(m_dice, 4)],
        'M_iou': [round(m_iou, 4)],
        'WFM': [round(wfm, 4)],
        'Sm': [round(sm, 4)],
        'Em': [round(em, 4)],
        'MAE': [round(MAE, 4)],
        'maxF': [round(maxf, 4)],
        'meanF': [round(maxf, 4)],
        'Recall':[round(recall_result, 4)],
        'Acc': [round(acc, 4)]
    }
    temp_df = pd.DataFrame(data)
    result_df = pd.concat([result_df, temp_df], ignore_index=True)

    # 保存DataFrame到Excel文件

excel_file = './result_map/results.csv'

result_df.to_csv(excel_file, index=False)
