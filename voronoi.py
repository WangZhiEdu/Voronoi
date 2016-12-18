import numpy as np
import scipy as sp
import scipy.spatial
import matplotlib.pyplot as plt
import math
from functools import cmp_to_key


class BoundVoronoi:
    def __init__(self, bound=(-35, 35, -35, 35), space=0.4):
        self._bounds = np.array(bound)
        self._space = space

    def _generate_by_points(self, points):
        # create the reflection symmetries points
        points_left = np.copy(points)
        points_left[:, 0] = self._bounds[0] - (points_left[:, 0] - self._bounds[0])
        points_right = np.copy(points)
        points_right[:, 0] = self._bounds[1] + (self._bounds[1] - points_right[:, 0])
        points_down = np.copy(points)
        points_down[:, 1] = self._bounds[2] - (points_down[:, 1] - self._bounds[2])
        points_up = np.copy(points)
        points_up[:, 1] = self._bounds[3] + (self._bounds[3] - points_up[:, 1])
        all_points = np.concatenate((points, points_left, points_right, points_down, points_up))
        # generate by scipy.spatial
        vor = sp.spatial.Voronoi(all_points)
        return vor

    def _read_poisson(self, file_name='poisson/Poisson.txt'):
        position_list = []
        with open(file_name, 'r', encoding='utf-8') as f:
            first_line = True
            for line in f:
                if first_line:
                    first_line = False
                    continue
                line_token = line.split(' ')
                position_list.append((float(line_token[0]), float(line_token[1])))
        # sort by coordinate
        position_list.sort(key=cmp_to_key(lambda a, b: a[0]-b[0] if a[0] != b[0] else a[1]-b[1]))
        return position_list

    def _move_point(self, centroid, point):
        gap_x = point[0] - centroid[0]
        gap_y = point[1] - centroid[1]
        distance = math.sqrt(gap_y**2 + gap_x**2)
        new_y = point[1] - self._space / distance * gap_y
        new_x = point[0] - self._space / distance * gap_x
        return np.array([new_x, new_y])



    def generate(self):
        # read Poisson.txt
        points = self._read_poisson()
        points = np.array(points)
        # extends
        x_width = self._bounds[1] - self._bounds[0]
        y_width = self._bounds[3] - self._bounds[2]
        points[:, 0] = points[:, 0] * x_width + self._bounds[0]
        points[:, 1] = points[:, 1] * y_width + self._bounds[2]
        vor = self._generate_by_points(points)
        sp.spatial.voronoi_plot_2d(vor)
        # plt.show()
        plt.savefig('images/vor_full.png')
        plt.close()
        # get the region of points
        vertice_list = []
        for i in range(len(points)):
            point = points[i]
            vertices = []
            region_index = vor.point_region[i]
            vertice_index_list = vor.regions[region_index]
            if len(vertice_index_list) == 0 \
                    or -1 in vertice_index_list:
                raise Exception('Region should not at boundary')
            for vertice_index in vertice_index_list:
                vertice = vor.vertices[vertice_index]
                vertices.append(vertice)
            vertices = np.array(vertices)
            vertice_list.append(vertices)
        # paint vertice_list
        fig = plt.figure()
        ax = fig.gca()
        for vertices in vertice_list:
            ax.plot(vertices[:, 0], vertices[:, 1], 'k-')
            end_line_x = [vertices[-1, 0], vertices[0, 0]]
            end_line_y = [vertices[-1, 1], vertices[0, 1]]
            ax.plot(end_line_x, end_line_y, 'k-')
        plt.savefig('images/vor.png')
        plt.close()
        # paint vertice
        space_vertice_list = []
        for i in range(len(points)):
            point = points[i]
            vertices = vertice_list[i]
            space_vertices = []
            for vertice in vertices:
                space_vertices.append(self._move_point(point, vertice))
            space_vertices = np.array(space_vertices)
            space_vertice_list.append(space_vertices)
        # paint space vertice
        fig = plt.figure()
        ax = fig.gca()
        for vertices in space_vertice_list:
            ax.plot(vertices[:, 0], vertices[:, 1], 'k-')
            end_line_x = [vertices[-1, 0], vertices[0, 0]]
            end_line_y = [vertices[-1, 1], vertices[0, 1]]
            ax.plot(end_line_x, end_line_y, 'k-')
        plt.savefig('images/vor_space.png')
        plt.close()
        # save points
        with open('Point.txt', 'w', encoding='utf-8') as f:
            f.write('NumPoints = %d\n' % len(space_vertice_list))
            for vertices in space_vertice_list:
                for vertice in vertices:
                    f.write('%f, %f \t' % (vertice[0], vertice[1]))
                f.write('\n')
        return vor, space_vertice_list


bound_vor = BoundVoronoi()
vor, vertice_list = bound_vor.generate()