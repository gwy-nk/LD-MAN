from torch.utils.data import Dataset
import pickle
from collections import Counter

from sklearn.model_selection import train_test_split
import numpy as np
from gensim import corpora
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import sklearn.metrics as sk

import torch.nn.functional as F
import torch
import pickle
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

import gc
dict_len = 163144


# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Text-Classification/blob/master/train.py


def load_News_mine(options):
    """
    # the visual features are just the average or max of all the global features
    the text is not hierarchical
    :param options:
    :return:
    """
    data_path = options['data_path']
    dict_file = options['dict_file']
    embedding_file = options['embedding_file']
    store_file = options['store_embedding_mat']
    emb_dim = options['emb_dim']
    img_global_path = options['img_global_path']
    max_text_len = options['max_len']
    img_scene_path = options['img_scene_path']
    img_num = options['img_num']

    sen_limit = options['sen_limit']
    word_limit = options['word_limit']

    class MUL_News_mul(Dataset):
        def __init__(self, text, v_global, v_scene, img_pos, labels, sen_num, word_num, img_dis):
            self.labels = labels
            self.text = text
            self.v_global = v_global
            self.v_scene = v_scene
            self.img_pos = img_pos
            self.sen_num = sen_num
            self.word_num = word_num
            self.img_dis = img_dis

        def __getitem__(self, idx):
            return [
                        self.text[idx, :],
                        self.v_global[idx, :, :],
                        self.v_scene[idx, :, :],

                        self.img_pos[idx, :],
                        self.sen_num[idx],
                        self.word_num[idx, :],
                        self.img_dis[idx, :, :],
                        self.labels[idx]
                    ]

        def __len__(self):
            return self.labels.shape[0]

        def shuffle_data(self, seed=123456):
            """
            shuffle the training data
            :param seed: sample seed
            :return:
            """
            random_index = np.random.RandomState(seed=seed).permutation(self.text.shape[0])

            self.labels = self.labels[random_index]
            self.text = self.text[random_index]
            self.v_global = self.v_global[random_index]
            self.v_scene = self.v_scene[random_index]
            self.img_pos = self.img_pos[random_index]
            self.sen_num = self.sen_num[random_index]
            self.word_num = self.word_num[random_index]
            self.img_dis = self.img_dis[random_index]

    def conduct_embedding_matrix(dict_file, embedding_file, store_file, emb_dim):
        import os
        if os.path.isfile(store_file):
            with open(store_file, 'rb') as file:
                embedding_matrix = pickle.load(file)

            return embedding_matrix
        else:

            """From blog.keras.io"""
            embeddings_index = {}
            f = open(embedding_file, 'r', encoding='UTF-8')

            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype=np.float32)
                embeddings_index[word] = coefs
            f.close()

            word_dict = corpora.Dictionary.load(dict_file)

            word_num = len(word_dict)

            embedding_mat = np.zeros((word_num, emb_dim), dtype=np.float32)
            for i, word in word_dict.items():
                embedding_v = embeddings_index.get(word)
                if embedding_v is not None:
                    embedding_mat[i] = embedding_v
            pretrained_wemb = np.concatenate((np.zeros(shape=(1, emb_dim)), embedding_mat))
            print(np.sum(pretrained_wemb))
            with open(store_file, 'wb+') as file:
                pickle.dump(file=file, obj=pretrained_wemb)

            return pretrained_wemb

    def pad_batch(one_news_data):

        _sen_num = min(len(one_news_data['all_text_ids']), sen_limit)
        _word_num = np.zeros(sen_limit, dtype=np.int)
        _text_id = np.zeros((sen_limit, word_limit), dtype=np.int)

        for i in range(sen_limit):
            for j in range(word_limit):
                try:
                    _text_id[i, j] = one_news_data['all_text_ids'][i][j] + 1
                except:
                    pass
            try:
                _word_num[i] = min(len(one_news_data['all_text_ids'][i]), word_limit)
            except:
                pass

        matrix_dis = np.zeros((img_num, sen_limit), dtype=np.int)
        img_poses = one_news_data['img_poses']
        for i in range(img_num):
            for j in range(sen_limit):
                try:
                    if j < _sen_num:
                        if img_poses[i] > j:
                            matrix_dis[i, j] = img_poses[i] - j
                        else:
                            matrix_dis[i, j] = j - img_poses[i] + 1
                except:
                    pass

        return _sen_num, _word_num, matrix_dis, _text_id

    def extract_features(data_dict, news_ids, labels):
        # pretrained_wemb = conduct_embedding_matrix(dict_file, embedding_file, store_file, emb_dim)
        ret_text = []
        ret_v_global = []
        ret_v_scene = []
        ret_v_poses = []
        ret_labels = []
        sen_num = []
        word_num = []
        img_dis = []

        img_feat_dict_global = pickle.load(open(img_global_path, 'rb'))
        img_feat_dict_scene = pickle.load(open(img_scene_path, 'rb'))

        for index in range(len(news_ids)):
            news_id = news_ids[index]
            label = labels[index]
            assert label == data_dict[news_id]['labels']

            _sen_num, _word_num, matrix_dis, _text_id = pad_batch(data_dict[news_id])
            sen_num.append(_sen_num)
            word_num.append(_word_num)
            img_dis.append(matrix_dis)
            ret_text.append(_text_id)

            temp_img_feature_g = []
            temp_img_feature_s = []
            temp_img_feature_p = []

            for i_, img_id in enumerate(data_dict[news_id]['img_ids']):
                temp_img_feature_g.append(img_feat_dict_global[img_id])
                temp_img_feature_s.append(img_feat_dict_scene[img_id])
                temp_img_feature_p.append(data_dict[news_id]['img_poses'][i_])

            if len(temp_img_feature_g) <= img_num:
                temp_img_feature_g = temp_img_feature_g + [np.zeros(2048) for _ in
                                                           range(img_num-len(temp_img_feature_g))]
                temp_img_feature_s = temp_img_feature_s + [np.zeros(2048) for _ in
                                                           range(img_num - len(temp_img_feature_s))]
                temp_img_feature_p = temp_img_feature_p + [0 for _ in
                                                           range(img_num - len(temp_img_feature_p))]
            else:
                temp_img_feature_g = temp_img_feature_g[0: img_num]
                temp_img_feature_s = temp_img_feature_s[0: img_num]
                temp_img_feature_p = temp_img_feature_p[0: img_num]

            ret_v_global.append(temp_img_feature_g)
            ret_v_scene.append(temp_img_feature_s)
            ret_v_poses.append(temp_img_feature_p)

            ret_labels.append(label)

        # ret_text = np.array(ret_text)
        ret_text = np.array(ret_text, dtype=np.int)
        ret_text = np.array(ret_text, dtype=np.int)

        ret_v_global = np.array(ret_v_global)
        ret_v_scene = np.array(ret_v_scene)
        ret_v_poses = np.array(ret_v_poses)

        ret_labels = np.array(ret_labels)

        sen_num = np.array(sen_num)

        word_num = np.array(word_num)
        img_dis = np.array(img_dis)

        return ret_text, ret_v_global, ret_v_scene, ret_v_poses, ret_labels, sen_num, word_num, img_dis

    all_data_dict = pickle.load(open(data_path, 'rb'))
    labels = []
    news_ids = []
    for news_id, news_content in all_data_dict.items():
        assert news_id == news_content['news_ids']
        news_ids.append(news_id)
        labels.append(news_content['labels'])

    ids_train, ids_test_val, label_train, label_test_val = train_test_split(news_ids, labels, test_size=0.2, random_state=0)
    ids_val, ids_test, label_val, label_test = train_test_split(ids_test_val, label_test_val, test_size=0.66, random_state=0)

    text_train, v_global_train, v_scene_train, v_poses_train, labels_train, sen_num_train, \
        word_num_train, img_dis_train = extract_features(all_data_dict, ids_train, label_train)
    text_val, v_global_val, v_scene_val, v_poses_val, labels_val, sen_num_val, word_num_val, img_dis_val\
        = extract_features(all_data_dict, ids_val, label_val)
    text_test, v_global_test, v_scene_test, v_poses_test, labels_test, sen_num_test, \
        word_num_test, img_dis_test = extract_features(all_data_dict, ids_test, label_test)

    print(text_train.size)
    print(labels_train.shape)
    # text, v_global, v_scene, v_object, img_pos, labels, sen_num, word_num, img_dis
    train_set = MUL_News_mul(text_train, v_global_train, v_scene_train, v_poses_train, labels_train,
                             sen_num_train, word_num_train, img_dis_train)
    val_set = MUL_News_mul(text_val, v_global_val, v_scene_val,  v_poses_val, labels_val, sen_num_val,
                           word_num_val, img_dis_val)
    test_set = MUL_News_mul(text_test, v_global_test, v_scene_test, v_poses_test, labels_test,
                            sen_num_test, word_num_test, img_dis_test)

    visual_dim = train_set[0][0].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = emb_dim
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (visual_dim, text_dim)

    pretrained_wemb = conduct_embedding_matrix(dict_file, embedding_file, store_file, emb_dim)

    return train_set, val_set, test_set, input_dims, pretrained_wemb


def preprocess_non_neural(modality, train_set, valid_set, test_set, pretrained_mat):
    # print(pretrained_mat[1])

    if modality == 'I':
        return train_set.visual, train_set.labels, test_set.visual, test_set.labels

    elif modality == 'T':
        with torch.no_grad():
            in_train = train_set.text
            # pretrained_mat = pretrained_mat
            x_train = F.embedding(torch.from_numpy(in_train), torch.from_numpy(pretrained_mat))
            x_train = x_train.numpy()
            x_train = np.mean(x_train, axis=1).squeeze()

            in_test = test_set.text
            # pretrained_mat = pretrained_mat
            x_test = F.embedding(torch.from_numpyin_test, torch.from_numpy(pretrained_mat))
            x_test = x_test.numpy()
            x_test = np.mean(x_test, axis=1).squeeze()

        del pretrained_mat
        gc.collect()

        return x_train, train_set.labels, x_test, test_set.labels

    else:
        with torch.no_grad():
            in_train = train_set.text
            # pretrained_mat = pretrained_mat
            x_train = F.embedding(torch.from_numpy(in_train), torch.from_numpy(pretrained_mat))
            x_train = x_train.numpy()
            x_train = np.mean(x_train, axis=1).squeeze()

            in_test = test_set.text
            # pretrained_mat = pretrained_mat
            x_test = F.embedding(torch.from_numpy(in_test), torch.from_numpy(pretrained_mat))
            x_test = x_test.numpy()
            x_test = np.mean(x_test, axis=1).squeeze()

        del pretrained_mat
        gc.collect()

        x_train = np.concatenate((x_train, train_set.visual), axis=1)
        x_test = np.concatenate((x_test, test_set.visual), axis=1)

        return x_train, train_set.labels, x_test, test_set.labels


def display(y_true, y_pred, result_file):

    # def filter_sentiment(y_true_l, y_pred_l, number_emotions):
    #
    #     y_pred_l_ret = []
    #     y_true_l_ret = []
    #
    #     for index in range(number_emotions):
    #         # y_true_l_ret.append(index)
    #         tmp_pred = []
    #         tmp_true = []
    #         for i in range(len(y_true_l)):
    #             if y_true_l[i] == index:
    #                 tmp_true.append(y_true_l[i])
    #                 tmp_pred.append(y_pred_l[i])
    #         y_pred_l_ret.append(tmp_pred)
    #         y_true_l_ret.append(tmp_true)
    #
    #     return y_true_l_ret, y_pred_l_ret
    #
    # print("==================================================================================================")
    # with open(result_file, 'wb+') as file:
    #     pickle.dump(obj=(y_pred, y_true), file=file)
    print("Accuracy on test set is {}".format(np.round(sk.accuracy_score(y_true, y_pred), 4)*100))

    # # print("F1 micro on test set is {}".format(sk.f1_score(y_true, y_pred, average='micro')))
    # print("F1 macro on test set is {}".format(sk.f1_score(y_true, y_pred, average='macro')))
    # print("F1 weighted on test set is {}".format(sk.f1_score(y_true, y_pred, average='weighted')))
    #
    # # print("precision micro on test set is {}".format(sk.precision_score(y_true, y_pred, average='micro')))
    # print("precision macro on test set is {}".format(sk.precision_score(y_true, y_pred, average='macro')))
    # print("precision weighted on test set is {}".format(sk.precision_score(y_true, y_pred, average='weighted')))
    #
    # # print("recall micro on test set is {}".format(sk.recall_score(y_true, y_pred, average='micro')))
    # print("recall macro on test set is {}".format(sk.recall_score(y_true, y_pred, average='macro')))
    # print("recall weighted on test set is {}".format(sk.recall_score(y_true, y_pred, average='weighted')))
    # print("---------------------------------------------------------------------------------------------------")
    #
    # RON_index_emotion = {
    #     0: "happy",
    #     1: "Sad",
    #     2: "Angry",
    #     3: "Don't Care",
    #     4: "Inspired",
    #     5: "Afraid",
    #     6: "Amused",
    #     7: "Annoyed"
    # }
    #
    # DMON_index_emotion = {
    #     0: "negative",
    #     1: "positive",
    #     2: "neutral",
    # }
    #
    # y_true_list = y_true.tolist()
    # y_pred_list = y_pred.tolist()
    #
    # y_true_set = set(y_true_list)
    # y_true_l_ret, y_pred_l_ret = filter_sentiment(y_true_list, y_pred_list, len(y_true_set))
    # if len(y_true_set) == 3:
    #     for index in range(3):
    #         print("Accuracy on {} is {}".format(DMON_index_emotion[index],
    #                                             np.round(sk.accuracy_score(np.array(y_true_l_ret[index]),
    #                                                               np.array(y_pred_l_ret[index])), 4)*100))
    # else:
    #     for index in range(8):
    #         print("Accuracy on {} is {}".format(RON_index_emotion[index],
    #                                             np.round(sk.accuracy_score(np.array(y_true_l_ret[index]),
    #                                                               np.array(y_pred_l_ret[index])), 4)*100))
    #
    # print("==================================================================================================")

    return np.round(sk.accuracy_score(y_true, y_pred), 4)*100

