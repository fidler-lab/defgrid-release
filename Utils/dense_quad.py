import os

import torch
import cv2
import random
import numpy as np
import copy
from tqdm import  tqdm
from collections import defaultdict

def create_init_cell_adjacent(n_row, n_column):
    n_row = n_row - 1
    n_column = (n_column - 1) * 2 #row column for cell
    cell_num = n_row * n_column
    adjacent = torch.zeros(cell_num, cell_num)
    for i in range(n_row):
        for j in range(n_column):
            idx = i * n_column + j
            #on the right
            if j != n_column - 1:
                adjacent[idx][idx + 1] = 1
            #on the left
            if j != 0:
                adjacent[idx][idx - 1] = 1
            #upper
            if i != 0:
                if i % 2 == 0 and ((j + 1) // 2) % 2 == 1:
                    adjacent[idx][idx - n_column] = 1
                if i % 2 == 1 and ((j + 1) // 2) % 2 == 0:
                    adjacent[idx][idx - n_column] = 1
            #below
            if i != n_row - 1:
                if i % 2 == 0 and ((j + 1) // 2) % 2 == 0:
                    adjacent[idx][idx + n_column] = 1
                if i % 2 == 1 and ((j + 1) // 2) % 2 == 1:
                    adjacent[idx][idx + n_column] = 1
    return adjacent

def create_init_cell_adjacent_jun(triange_2_point):

    # n_triangle = triange_2_point.shape[0]
    triange_2_point = triange_2_point.long().numpy()
    n_triangle = triange_2_point.shape[0]
    adjacent = torch.zeros(n_triangle, n_triangle, 3)
    for i_tri in tqdm(range(n_triangle)):
        for z in range(3):
            a = triange_2_point[i_tri][z]
            b = triange_2_point[i_tri][(z + 1) % 3]
            a_b = a * n_triangle + b
            # a_b = str(a) + '_' + str(b)
            find_idx = 0
            find = False
            for j_tri in range(n_triangle):
                if j_tri == i_tri:
                    continue
                for new_z in range(3):
                    c = triange_2_point[j_tri][new_z]
                    d = triange_2_point[j_tri][(new_z + 1) % 3]
                    c_d = c * n_triangle + d

                    d_c = d * n_triangle + c
                    if c_d == a_b or d_c == a_b:
                        find = True
                if find:
                    find_idx = j_tri
                    break
            if find:
                adjacent[i_tri][find_idx][z] = 1
    return adjacent

def create_init_cell_adjacent_jun_faster(triange_2_point):
    # n_triangle = triange_2_point.shape[0]
    triange_2_point = triange_2_point.long().numpy()
    n_triangle = triange_2_point.shape[0]
    adjacent = torch.zeros(n_triangle, n_triangle, 3)
    edge_index = dict()
    for tri_idx, tri in enumerate(triange_2_point):
        for i_edge in range(3):
            a = tri[i_edge]
            b = tri[(i_edge + 1) % 3]

            num1 = a * n_triangle + b
            num2 = b * n_triangle + a

            if num1 not in edge_index and num2 not in edge_index:
                edge_index[num1] = [[tri_idx, i_edge]]
            elif num1 in edge_index:
                edge_index[num1].append([tri_idx, i_edge])
            elif num2 in edge_index:
                edge_index[num2].append([tri_idx, i_edge])
            else:
                raise ValueError
    singular_edge = 0
    # import ipdb
    # ipdb.set_trace()
    for edge_num in edge_index.keys():
        all_tri = edge_index[edge_num]
        if len(all_tri) == 1:
            singular_edge += 1
        if len(all_tri) == 2:
            adjacent[all_tri[0][0]][all_tri[1][0]][all_tri[0][1]] = 1
            adjacent[all_tri[1][0]][all_tri[0][0]][all_tri[1][1]] = 1
    # import ipdb
    # ipdb.set_trace()
    return adjacent


def create_edge_idx(triangle_list, n_point):
    edge_face_dict = defaultdict(list)

    def get_edge_id(i, j, n):
        min_n = min(i, j)
        max_n = max(i, j)
        return int(min_n * n + max_n)

    for t_id, t in enumerate(triangle_list):
        a, b, c = t
        edge_id = get_edge_id(a, b, n_point)
        edge_face_dict[edge_id].append(t_id)

        edge_id = get_edge_id(a, c, n_point)
        edge_face_dict[edge_id].append(t_id)

        edge_id = get_edge_id(c, b, n_point)
        edge_face_dict[edge_id].append(t_id)

    edge_list = []
    all_edge_list = []
    edge_face_list= []
    skip_2 = 0
    for edge_id in edge_face_dict:
        a = edge_id % n_point
        b = edge_id / n_point
        all_edge_list.append([a, b])
        if len(edge_face_dict[edge_id]) < 2:
            skip_2 += 1
        else:
            edge_list.append([a, b])
            edge_face_list.append(edge_face_dict[edge_id])
    return np.asarray(edge_list), np.asarray(edge_face_list), np.asarray(all_edge_list)

def create_edge_adj(edge_list):
    n_edge = len(edge_list)
    adj = np.zeros((n_edge, n_edge))
    point_to_edge = defaultdict(list)
    for idx_e, e in enumerate(edge_list):
        a, b = e
        point_to_edge[a].append(idx_e)
        point_to_edge[b].append(idx_e)

    for p in point_to_edge.keys():
        all_e = point_to_edge[p]
        n_e = len(all_e)
        for i in range(n_e):
            for j in range(n_e):
                adj[all_e[i]][all_e[j]] = 1
                adj[all_e[j]][all_e[i]] = 1
    for i in range(n_edge):
        adj[i][i] = 0
    return adj

def create_init_point_adjacent(n_row, n_column):
    point_num = n_row * n_column
    adjacent = torch.zeros([point_num, point_num])
    #for the first part
    for i in range(n_row):
        for j in range(n_column):
            idx = i * n_column + j
            #next node on the right
            if j != n_column - 1:
                adjacent[idx][idx + 1] = 1
            #prev node on the left
            if j != 0:
                adjacent[idx][idx - 1] = 1
            #upper node:
            if i != 0:
                adjacent[idx][idx - n_column] = 1
            #below node:
            if i != n_row - 1:
                adjacent[idx][idx + n_column] = 1

            if (i + j) % 2 == 0: #odd-odd or even-even
                #upper left
                if i != 0 and j != 0:
                    adjacent[idx][idx - n_column - 1] = 1
                #upper right 
                if i != 0 and j != n_column - 1:
                    adjacent[idx][idx - n_column + 1] = 1
                #below left
                if i != n_row - 1 and j != 0:
                    adjacent[idx][idx + n_column - 1] = 1
                #below right
                if i != n_row - 1 and j != n_column - 1:
                    adjacent[idx][idx + n_column + 1] = 1

    return adjacent.float()

def create_initial_triangle(n_row, n_column):
    # assume we have a hxw grid
    # they are originized as triangles
    # what's their triangle indices?
    triangle = []
    area_mask = []
    for i in range(n_row - 1):
        for j in range(n_column - 1):
            a = i * n_column + j
            b = i * n_column + j + 1
            c = (i + 1) * n_column + j
            d = (i + 1) * n_column + j + 1
            if (i + j) % 2 == 0: #odd-odd or even-even
                triangle.append([a, c, d])
                triangle.append([a, d, b])
            else: #odd-even or even-odd
                triangle.append([a, c, b])
                triangle.append([b, c, d])
            area_mask.append(1)
            area_mask.append(1)
    return torch.tensor(triangle).float(), torch.tensor(area_mask).float()

def create_init_point_mask(n_row, n_column):
    # for a  3*4, 3 is row, 4 is column
    # for the last dimension, the first [..., 0] is x, the second [..., 1] is y
    # print('Init Grid')
    point_num = n_row * n_column
    init_mask = torch.ones([point_num, 2])

    for i in range(n_row):
        idx = i * n_column + 0
        init_mask[idx, 0] = 0
        idx = i * n_column + n_column - 1
        init_mask[idx, 0] = 0
    for i in range(n_column):
        idx = 0 + i
        init_mask[idx, 1] = 0
        idx = (n_row - 1) * n_column + i
        init_mask[idx, 1] = 0

    return init_mask.float()

def create_init_point(n_row, n_column, scale_pos=False, img_size=None):
    x = torch.linspace(0, 1, n_column).reshape(1, n_column, 1).repeat(n_row, 1, 1)
    y = torch.linspace(0, 1, n_row).reshape(n_row, 1, 1).repeat(1, n_column, 1)

    # TODO: adjust this such that the maximum height is 1 and scale the second one
    if scale_pos:
        max_height = max(img_size[0], img_size[1]) # scale the grid according to the image shape
        x = x * img_size[1] / max_height
        y = y * img_size[0] / max_height
    init_point = torch.cat((x, y), dim=-1).reshape(-1, 2)
    return init_point.float()