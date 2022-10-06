from __future__ import print_function
from release_utils import load_News_mine, display
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import os, sys
import argparse
import torch
import torch.nn as nn
import numpy as np

import os
import xlrd

import xlwt

from xlutils.copy import copy


def excelwrite(L=None, file_name=None):

    import os
    if not os.path.isfile(file_name):
        xls = xlwt.Workbook()
        sht1 = xls.add_sheet('Sheet1')
        sht1.write(0, 0, "lamda_img")
        sht1.write(0, 1, "lamda_pos")
        sht1.write(0, 2, "lamda_mul")
        sht1.write(0, 3, "Test Acc")
        xls.save(file_name)

    workbook = xlrd.open_workbook(file_name, formatting_info=True)

    sheet = workbook.sheet_by_index(0)

    rowNum = sheet.nrows
    newbook = copy(workbook)
    newsheet = newbook.get_sheet(0)

    newsheet.write(rowNum, 0, str(L[0]))
    newsheet.write(rowNum, 1, str(L[1]))
    newsheet.write(rowNum, 2, str(L[2]))
    newsheet.write(rowNum, 3, str(L[3]))

    # 覆盖保存
    newbook.save(file_name)


def print_paras(options):
    print('-----------------------')
    for key, value in options.items():
        print(key, value)
    print('-----------------------')


def build_HiMAN(pretrained_mat, options):
    from models.LD_MAN import HiMAN
    # from models.img_att import Model
    import torch.optim as optim

    model = HiMAN(
        n_classes=options['output_dim'],
        vocab_size=pretrained_mat.shape[0],
        emb_size=pretrained_mat.shape[1],
        word_rnn_size=options['rnn_size'],
        sentence_rnn_size=options['rnn_size'],
        word_rnn_layers=1,
        sentence_rnn_layers=1,
        word_att_size=100,
        sentence_att_size=100,
        dropout=options['dropout'],
        pretrained_mat=pretrained_mat,
        options=options,
        img_global_size=options['img_global_size']
    )

    loss_fn = nn.CrossEntropyLoss(size_average=False)
    optim = optim.Adam(model.parameters(), lr=options['learning_rate'])

    return model, loss_fn, optim


def main(options):
    DTYPE = torch.FloatTensor

    # parse the input args
    run_id = options['run_id']
    epochs = options['epochs']
    ckpt_path = options['ckpt_path']
    result_path = options['result_path']
    dataset = options['dataset']
    output_dim = options['output_dim']
    batch_sz = options['batch_sz']
    train_method = options['train_method']
    sen_limit = options['sen_limit']
    word_limit = options['word_limit']
    img_limit = options['img_num']

    print("Training initializing... Setup ID is: {}".format(run_id))

    # prepare the paths for storing models and outputs
    # store the ckpts
    ckpt_path = os.path.join(
        ckpt_path, "model_{}.pth".format(dataset))
    # store the predicted labels
    # result_path = os.path.join(
    #     result_path, "results_{}_{}.pkl".format(dataset))
    print("Temp location for model ckpts: {}".format(ckpt_path))
    # print("Predicting results are in: {}".format(result_path))

    train_set, valid_set, test_set, input_dims, pretrained_mat = load_News_mine(options)

    model, loss_fn, optim = build_HiMAN(pretrained_mat, options)

    test_iterator = DataLoader(test_set, batch_size=batch_sz, shuffle=False)

    # best_model = torch.load(ckpt_path)
    # best_model.eval()
    best_model = torch.load(ckpt_path)
    states = best_model.state_dict()
    model.load_state_dict(states)
    best_model = model
    best_model.eval()
    best_model = model.cuda()

    predict_labels = []
    truth_labels = []

    for batch in test_iterator:
        x = batch[:-1]
        x_t = Variable(x[0].view(-1, sen_limit, word_limit), requires_grad=False).cuda()
        x_v_g = Variable(x[1].view(-1, img_limit, 2048).float().type(DTYPE), requires_grad=False).squeeze().cuda()
        x_v_s = Variable(x[2].view(-1, img_limit, 2048).float().type(DTYPE), requires_grad=False).squeeze().cuda()
        x_v_p = Variable(x[3].view(-1, img_limit).float().type(torch.LongTensor),
                         requires_grad=False).squeeze().cuda()
        x_sen_num = Variable(x[4].view(-1).float().type(DTYPE), requires_grad=False).squeeze().cuda()
        x_word_num = Variable(x[5].view(-1, sen_limit).float().type(DTYPE), requires_grad=False).squeeze().cuda()
        x_img_dis = Variable(x[6].view(-1, img_limit, sen_limit).float().type(DTYPE),
                             requires_grad=False).squeeze().cuda()

        y = Variable(batch[-1]).cuda()

        if x[0].shape[0] == 1:
            pass
        else:
            output_test = best_model(x_t, x_sen_num, x_word_num, x_v_g, x_v_s, None, x_v_p, x_img_dis)
            output_test = output_test.cpu().data.numpy().reshape(-1, output_dim)
            predict_labels += np.argmax(output_test, axis=1).squeeze().tolist()
            y = y.cpu().data.numpy()
            truth_labels += y.squeeze().tolist()

    # these are the needed metrics
    all_true_label = np.array(truth_labels)
    all_predicted_label = np.array(predict_labels)

    best_Acc = display(all_true_label, all_predicted_label, result_path)

    return best_Acc



if __name__ == "__main__":

    preprocess_data_path = "data"
    glove_path = "glove"

    data_path = {
        "RON": preprocess_data_path + "/RON/data_RON.pkl",
        "DMON": preprocess_data_path + "/DMON/data_DMON.pkl"
    }
    dict_file = {
        "RON": preprocess_data_path + "/RON/word_id.dict",
        "DMON": preprocess_data_path + "/DMON/word_id.dict"
    }
    store_embedding_mat = {
        "RON": preprocess_data_path + "/RON/embedding_mat.pkl",
        "DMON": preprocess_data_path + "/DMON/embedding_mat.pkl"
    }
    img_global_path = {
        "RON": preprocess_data_path + "/RON/global_resnet152.pkl",
        "DMON": preprocess_data_path + "/DMON/global_resnet152.pkl"
    }

    img_scene_path = {
        "RON": preprocess_data_path + "/RON/RON_scene_resnet152.pkl",
        "DMON": preprocess_data_path + "/DMON/DMON_scene_resnet152.pkl"
    }

    output_dim = {
        "RON": 8,
        "DMON": 3
    }

    img_num = {
        "RON": 3,
        "DMON": 6
    }
    sen_limit = {
        "RON": 30,
        "DMON": 32
    }
    word_limit = {
        "RON": 20,
        "DMON": 20
    }

    OPTIONS = argparse.ArgumentParser()

    # some paths
    OPTIONS.add_argument('--embedding_file', dest='embedding_file', type=str,
                         default=glove_path + '/glove.6B.300d.txt')
    OPTIONS.add_argument('--ckpt_path', dest='ckpt_path',
                         type=str, default='ckpt')
    OPTIONS.add_argument('--result_path', dest='result_path',
                         type=str, default='results')
    OPTIONS.add_argument('--log_path', dest='log_path',
                         type=str, default='logs')

    # basic parameters
    OPTIONS.add_argument('--run_id', dest='run_id', type=str, default='RON_test_img_att')
    OPTIONS.add_argument('--excel_name', dest='excel_name', type=str, default='RON_log_')
    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=10)
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=20)
    OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=True)
    OPTIONS.add_argument('--gpu', dest='gpu', type=str, default='0')
    OPTIONS.add_argument('--dataset', dest='dataset', type=str, default="DMON")

    OPTIONS.add_argument('--batch_sz', dest='batch_sz', type=int, default=64)
    OPTIONS.add_argument('--train_method', dest='train_method', type=str, default="HiMAN",
                         choices=['HiMAN'])

    OPTIONS.add_argument('--max_len', dest='max_len', type=int, default=2400)
    OPTIONS.add_argument('--emb_dim', dest='emb_dim', type=int, default=300)
    OPTIONS.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001)
    OPTIONS.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.0005)

    OPTIONS.add_argument('--rnn_size', dest='rnn_size', type=int, default=256)
    OPTIONS.add_argument('--hidden', dest='hidden', type=int, default=256)
    OPTIONS.add_argument('--mid_hidden', dest='mid_hidden', type=int, default=512)
    OPTIONS.add_argument('--img_global_size', dest='img_global_size', type=int, default=2048)
    OPTIONS.add_argument('--img_lamda', dest='img_lamda', type=float, default=0.0)
    OPTIONS.add_argument('--pos_lamda', dest='pos_lamda', type=float, default=0.5)
    OPTIONS.add_argument('--mul_lamda', dest='mul_lamda', type=float, default=1.0)
    OPTIONS.add_argument('--dropout', dest='dropout', type=float, default=0.05)

    OPTIONS.add_argument('--n_head', dest='n_head', type=int, default=8)
    OPTIONS.add_argument('--hidden_size_head', dest='hidden_size_head', type=int, default=32)

    # Add supp RON dataset
    OPTIONS.add_argument('--img_sum_methods', dest='img_sum_methods', type=str, default="max")

    PARAMS = vars(OPTIONS.parse_args())

    os.environ['CUDA_VISIBLE_DEVICES'] = PARAMS['gpu']

    PARAMS['data_path'] = data_path[PARAMS['dataset']]
    PARAMS['dict_file'] = dict_file[PARAMS['dataset']]
    PARAMS['store_embedding_mat'] = store_embedding_mat[PARAMS['dataset']]
    PARAMS['img_global_path'] = img_global_path[PARAMS['dataset']]
    PARAMS['output_dim'] = output_dim[PARAMS['dataset']]
    PARAMS['img_num'] = img_num[PARAMS['dataset']]
    PARAMS['img_scene_path'] = img_scene_path[PARAMS['dataset']]
    PARAMS['sen_limit'] = sen_limit[PARAMS['dataset']]
    PARAMS['word_limit'] = word_limit[PARAMS['dataset']]

    print_paras(PARAMS)
    # # DMON
    if PARAMS['dataset'] == 'DMON':
        PARAMS['img_lamda'] = 1
        PARAMS['pos_lamda'] = 1
        PARAMS['mul_lamda'] = 2

    # RON
    if PARAMS['dataset'] == 'RON':
        PARAMS['img_lamda'] = 1
        PARAMS['pos_lamda'] = 10
        PARAMS['mul_lamda'] = 10

    acc = main(PARAMS)
    print(acc)
