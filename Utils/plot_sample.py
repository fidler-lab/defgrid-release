import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from skimage import draw
import torch
import cv2
from matplotlib.path import Path

"""
plot with adjacency matrix
"""

def plot_colored_image_with_condition(image, condition, save_path, plot_color=True, image_weight=0.65, return_fig=False):

    index_num = condition.max() + 1
    color_mapping = torch.randint(0, 256, size=(index_num, 3))
    color_mask = color_mapping[condition.long()]

    overlay_image = (image * 255).astype(np.uint8)
    overlay_color = color_mask.cpu().numpy().astype(np.uint8)
    overlay = cv2.addWeighted(overlay_image, image_weight, overlay_color, 1 - image_weight, 0)
    if not plot_color:
        cv2.imwrite(save_path, overlay[:, :, ::-1])
    else:
        plt.subplot(121)
        plt.axis('off')
        plt.imshow(overlay)
        plt.subplot(122)
        plt.axis('off')
        plt.imshow(overlay_color)
        plt.savefig(save_path)
        plt.close()
    if return_fig:
        return overlay

def plot_points(p, save_name=None):
    plt.scatter(p[:,0], p[:1])
    plt.savefig(save_name)
    plt.close()

def plot_deformed_lattice_on_image_with_affinity_label(lattice_pos, ori_image, adjacent, save_path, pairs, affinity, grid_pos, thresh, mask=None):
    
    plt.gca().invert_yaxis()
    image = ori_image.copy()
    #for edges
    for idx, p in enumerate(pairs):
        
        x1, y1 = grid_pos[p[0]]
        x2, y2 = grid_pos[p[1]]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #line = draw.line(x1, y1, x2, y2)
        line = draw.line(y1, x1, y2, x2)
        pred = affinity[idx]
        if pred > thresh:
            image[line] = 0, 1, 0
        else:
            image[line] = 1, 1, 0

    fig, ax = plt.subplots(figsize=(image.shape[1]//30, image.shape[0]//30))
    ax.imshow(image)

    #plot control points
    x,y = np.where(adjacent!=0)
    conn = zip(x,y)

    if not mask is None:
        color = ['r' if i == 1 else 'g' for i in mask]
        size = [4 if i == 1 else 1 for i in mask ]
    else:
        color = ['r' for i in range(lattice_pos.shape[0])]
        size = [1 for i in range(lattice_pos.shape[0])]

    graph = nx.Graph()
    #adding nodes/connections in the graph
    for node in range(lattice_pos.shape[0]):
        graph.add_node(node)

    graph.add_edges_from(conn)

    nx.draw(graph, [(x,y) for x,y in lattice_pos], node_size=size, ax=ax, node_color=color)

    plt.savefig(save_path)
    plt.close()

def plot_deformed_lattice_on_image_with_class_label(lattice_pos, ori_image, adjacent, save_path, class_label, grid_pos, mask=None):
    
    plt.gca().invert_yaxis()
    image = ori_image.copy()

    fig, ax = plt.subplots(figsize=(image.shape[1]//30, image.shape[0]//30))
    ax.imshow(image)

    #plot control points
    x,y = np.where(adjacent!=0)
    conn = zip(x,y)

    if not mask is None:
        color = ['r' if i == 1 else 'g' for i in mask]
        size = [4 if i == 1 else 1 for i in mask ]
    else:
        color = ['r' for i in range(lattice_pos.shape[0])]
        size = [1 for i in range(lattice_pos.shape[0])]

    graph = nx.Graph()
    #adding nodes/connections in the graph
    for node in range(lattice_pos.shape[0]):
        graph.add_node(node)

    graph.add_edges_from(conn)

    nx.draw(graph, [(x,y) for x,y in lattice_pos], node_size=size, ax=ax, node_color=color)

    for i in range(grid_pos.shape[0]):
        x, y = grid_pos[i]
        plt.text(x, y, class_label[i], color='g')
        
    plt.savefig(save_path)
    plt.close()

def plot_deformed_lattice_on_image(lattice_pos, image, adjacent, save_path=None, grid_pairs=None, triangles=None, mask=None,
                                   return_fig=False, boundary_p=None, obj_mask=None, matching_point=None):
    fig, ax = plt.subplots(figsize=(image.shape[1]//20, image.shape[0]//20))
    ax.imshow(image)
    #for affinity merge
    if not grid_pairs is None:
        #N x 2
        point_pairs = pairs_grid2point(grid_pairs, triangles)
        point_num = adjacent.shape[0]
        adjacent_idx = point_pairs[:, 0] * point_num + point_pairs[:, 1]
        adjacent = adjacent.reshape(-1)
        adjacent[adjacent_idx] = 0
        adjacent = adjacent.reshape(point_num, point_num)       
        adjacent = adjacent - np.tril(adjacent) #since pairs only consist the unidirectional half
    #for upsample
    if not mask is None:
        color = ['r' if i == 1 else 'g' for i in mask]
        size = [4 if i == 1 else 1 for i in mask ]
    else:
        color = ['r' for i in range(lattice_pos.shape[0])]
        size = [1 for i in range(lattice_pos.shape[0])]
    x,y = np.where(adjacent!=0)
    conn = zip(x,y)

    graph = nx.Graph()
    #adding nodes/connections in the graph
    for node in range(lattice_pos.shape[0]):
        graph.add_node(node)

    graph.add_edges_from(conn)

    nx.draw(graph, [(x,y) for x,y in lattice_pos], node_size=size, ax=ax, node_color=color)

    if not boundary_p is None:
        for idx, line in enumerate(boundary_p):
            new_l = np.zeros((line.shape[0] + 1, line.shape[1]))
            new_l[:line.shape[0]] = line
            new_l[-1] = line[0]
            # new_l = np.stack((line, line[0][np.newaxis]), axis=0)
            ax.plot(new_l[:, 0], new_l[:, 1], '-*', color='C' + str(idx))
    if not matching_point is None:
        if np.ndim(matching_point[0]) == 2:
            ax.plot(matching_point[0][:, 0],matching_point[0][:, 1], 'b')
            ax.plot(matching_point[1][:, 0], matching_point[1][:, 1], 'g')

            for i in range(matching_point[0].shape[0]):
                ax.plot([matching_point[0][i, 0], matching_point[1][i, 0]],
                    [matching_point[0][i, 1], matching_point[1][i, 1]], 'r')
        else:
            ax.plot([matching_point[0][0], matching_point[1][0]],
                    [matching_point[0][1], matching_point[1][1]], 'r')
    if not obj_mask is None:
        # Apply the mask on top
        ax.imshow(obj_mask, alpha=0.2, cmap="cool")
    if not save_path is None:
        plt.savefig(save_path)


    if return_fig:
        plt.close()
        return fig
    plt.close()


def plot_boundary_on_image(line_list, image, save_path=None,  return_fig=False):
    fig, ax = plt.subplots(figsize=(image.shape[1]//30, image.shape[0]//30))
    ax.imshow(image)

    for idx, line in enumerate(line_list):
        ax.plot(line[:, 0], line[:, 1], '-*', color='C'+str(idx))
        
    if not save_path is None:
        plt.savefig(save_path)
    if return_fig:
        plt.close()
        return fig

def plot_deformed_lattice_on_image_with_boundary_p(lattice_pos, image, adjacent, save_path=None, gt_points=None, sampled_point=None,
                                                   grid_pairs=None, triangles=None, mask=None, return_fig=False):

    fig, ax = plt.subplots(figsize=(image.shape[1]//20, image.shape[0]//20))
    ax.imshow(image)
    #for affinity merge
    if not grid_pairs is None:
        #N x 2
        point_pairs = pairs_grid2point(grid_pairs, triangles)
        point_num = adjacent.shape[0]
        adjacent_idx = point_pairs[:, 0] * point_num + point_pairs[:, 1]
        adjacent = adjacent.reshape(-1)
        adjacent[adjacent_idx] = 0
        adjacent = adjacent.reshape(point_num, point_num)
        adjacent = adjacent - np.tril(adjacent) #since pairs only consist the unidirectional half
    #for upsample
    if not mask is None:
        color = ['r' if i == 1 else 'g' for i in mask]
        size = [4 if i == 1 else 1 for i in mask ]
    else:
        color = ['r' for i in range(lattice_pos.shape[0])]
        size = [1 for i in range(lattice_pos.shape[0])]
    x,y = np.where(adjacent!=0)
    conn = zip(x,y)

    graph = nx.Graph()
    #adding nodes/connections in the graph
    for node in range(lattice_pos.shape[0]):
        graph.add_node(node)

    graph.add_edges_from(conn)

    nx.draw(graph, [(x,y) for x,y in lattice_pos], node_size=size, ax=ax, node_color=color)
    if not gt_points is None:
        ax.scatter(gt_points[:,0], gt_points[:,1])
    if not sampled_point is None:
        ax.scatter(sampled_point[:,0], sampled_point[:,1])
    if not save_path is None:
        plt.savefig(save_path)
        plt.close()
    if return_fig:
        return fig


def plot_boundary_p(gt_points=None, sampled_point=None,
                    return_fig=False, save_path=None, size=224):
    fig, ax = plt.subplots(figsize=(20, 20))
    if not gt_points is None:
        ax.scatter(gt_points[:, 0], gt_points[:, 1])
    if not sampled_point is None:
        ax.scatter(sampled_point[:, 0], sampled_point[:,1])

    ax.set_xlim([0, size])
    ax.set_ylim([size, 0])
    # axes.set_xlim([xmin, xmax])
    # axes.set_ylim([ymin, ymax])
    # axes.set_ylim([ymin, ymax])
    if return_fig:
        return fig
    else:
        plt.savefig(save_path)
        plt.close()


def plot_matching(gt_points=None, pred_point=None, matching_point=None,
                    return_fig=False, save_path=None, size=224):
    fig, ax = plt.subplots(figsize=(20, 20))

    ax.plot(gt_points[:, 0], gt_points[:, 1], 'b')
    ax.plot(pred_point[:, 0], pred_point[:, 1], 'g')

    for i in range(pred_point.shape[0]):
        ax.plot([pred_point[i, 0], matching_point[i, 0]],
                [pred_point[i, 1], matching_point[i, 1]], 'r')
    ax.set_xlim([0, size])
    ax.set_ylim([size, 0])
    # axes.set_xlim([xmin, xmax])
    # axes.set_ylim([ymin, ymax])
    # axes.set_ylim([ymin, ymax])
    if return_fig:
        return fig
    else:
        plt.savefig(save_path)
        plt.close()



def imdrawcontour(pointsnp, linecolor, pointcolor, ima):
    height, width, _ = ima.shape
    pnum = pointsnp.shape[0]
    for i in range(pnum):
        pbe = pointsnp[i]
        pen = pointsnp[(i + 1) % pnum]
        pbe = np.round(pbe * height).astype(np.int32)
        pen = np.round(pen * height).astype(np.int32)
        if np.any(pbe < 0):
            continue
        if np.any(pbe > height - 1):
            continue
        if np.any(pen < 0):
            continue
        if np.any(pen > height - 1):
            continue
        cv2.circle(ima, (pbe[0], pbe[1]), 2, pointcolor, thickness=-1)
        cv2.line(ima, (pbe[0], pbe[1]), (pen[0], pen[1]), linecolor, thickness=1)
        # print pbe[0], pbe[1]
        # cv2.imshow('p', ima)
        # cv2.waitKey()

    return ima



def plot_boundary_p_link(gt_points=None, sampled_point=None,
                    return_fig=False, save_path=None, size=224, circle=True):
    fig, ax = plt.subplots(figsize=(20, 20))
    if not gt_points is None:
        ax.scatter(gt_points[:, 0], gt_points[:, 1])
    if not sampled_point is None:
        if isinstance(sampled_point, list):
            for s in sampled_point:
                new_p = np.zeros((s.shape[0] + 1, s.shape[1]))
                new_p[:-1] = s
                new_p[-1] = s[0]
                ax.plot(new_p[:, 0], new_p[:, 1])
        else:
            s = sampled_point
            if circle:
                new_p = np.zeros((s.shape[0] + 1, s.shape[1]))
                new_p[:-1] = s
                new_p[-1] = s[0]
            else:
                new_p = s
            ax.plot(new_p[:, 0], new_p[:, 1])

    ax.set_xlim([0, size])
    ax.set_ylim([size, 0])
    if not save_path is None:
        plt.savefig(save_path)
    if return_fig:
        return fig

#convert grid pairs to point pairs
#grid_pairs N x 2 
#triangles: F x 3
def pairs_grid2point(grid_pairs, triangles):

    pairs_num = grid_pairs.shape[0]
    grid_pairs_triangle = triangles[grid_pairs.reshape(-1)].reshape(pairs_num, 2, 3)
    point_pairs = []
    #possibly using parallel
    for i in range(pairs_num):
        tmp = grid2point(grid_pairs_triangle[i])
        if len(tmp) != 2:
            raise ValueError('unadjacent pairs')
        point_pairs.append(grid2point(grid_pairs_triangle[i]))
    return np.array(point_pairs)

#pairs: 2x3
def grid2point(pairs):
    return np.intersect1d(pairs[0], pairs[1])

def polygon2mask(mask_shape, verts):
    x, y = np.meshgrid(np.arange(mask_shape[1]), np.arange(mask_shape[0]))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    path = Path(verts)
    grid = path.contains_points(points)
    grid = grid.reshape(mask_shape)
    return grid



def plot_normal_map(normal, save_name=None, return_fig=False):
    # import ipdb
    # ipdb.set_trace()
    X, Y = np.meshgrid(np.arange(0, 224), np.arange(0, 224))
    U = np.cos(normal)
    V = np.sin(normal)
    # import ipdb
    # ipdb.set_trace()
    fig3, ax3 = plt.subplots(figsize=(10,10))
    ax3.set_title("pivot='tip'; scales with x view")
    M = np.hypot(U, V)
    Q = ax3.quiver(X, Y, U, V, M, units='x', pivot='tip', width=10)
    qk = ax3.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                       coordinates='figure')
    # ax3.scatter(X, Y, color='0.5', s=1)

    if return_fig:
        return fig3
    else:
        plt.savefig(save_name)
        plt.close()
    