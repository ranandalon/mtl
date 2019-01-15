import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS


class Point():
    def __init__(self, x_dist, y_dist, row, col):
        self._x_dist = x_dist
        self._y_dist = y_dist
        self._row = row
        self._col = col
        self._x_center = -x_dist + col
        self._y_center = -y_dist + row
        #self._dist_to_center = np.sqrt((col - self._x_center)**2 + (row - self._y_center)**2)
        self.cd = None  # core distance
        self.rd = None  # reachability distance
        self.processed = False

    def distance(self, point):
        return np.sqrt((point._x_center - self._x_center)**2 + (point._y_center - self._y_center)**2)


def pre_process(x_y_mask):
    point_list = []
    y_dist = x_y_mask[:, :, 0]
    x_dist = x_y_mask[:, :, 1]
    mask = x_y_mask[:, :, 2]
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if mask[row, col] == 0:
                continue
            new_point = Point(x_dist[row][col], y_dist[row][col], row, col)
            point_list.append(new_point)
    return point_list


class Optics():
    def __init__(self, pts_list, min_cluster_size, epsilon):
        self.pts = pts_list
        self.min_cluster_size = min_cluster_size
        self.max_radius = epsilon

    def _setup(self):
        for p in self.pts:
            p.rd = None
            p.processed = False
        self.unprocessed = [p for p in self.pts]
        self.ordered = []

    def _core_distance(self, point, neighbors):
        if point.cd is not None:
            return point.cd
        if len(neighbors) >= self.min_cluster_size - 1:
            sorted_neighbors = sorted([n.distance(point) for n in neighbors])
            point.cd = sorted_neighbors[self.min_cluster_size - 2]
        return point.cd

    def _neighbors(self, point):
        return [p for p in self.pts if p is not point and p.distance(point) <= self.max_radius]

    def _processed(self, point):
        point.processed = True
        self.unprocessed.remove(point)
        self.ordered.append(point)

    def _update(self, neighbors, point, seeds):
        # for each of point's unprocessed neighbors n...
        for n in [n for n in neighbors if not n.processed]:
            # find new reachability distance new_rd
            # if rd is null, keep new_rd and add n to the seed list
            # otherwise if new_rd < old rd, update rd
            new_rd = max(point.cd, point.distance(n))
            if n.rd is None:
                n.rd = new_rd
                seeds.append(n)
            elif new_rd < n.rd:
                n.rd = new_rd

    def run(self):
        self._setup()
        # for each unprocessed point (p)...
        while self.unprocessed:
            point = self.unprocessed[0]
            # mark p as processed
            # find p's neighbors
            self._processed(point)
            point_neighbors = self._neighbors(point)
            # if p has a core_distance, i.e has min_cluster_size - 1 neighbors
            if self._core_distance(point, point_neighbors) is not None:
                # update reachability_distance for each unprocessed neighbor
                seeds = []
                self._update(point_neighbors, point, seeds)
                # as long as we have unprocessed neighbors...
                while (seeds):
                    # find the neighbor n with smallest reachability distance
                    seeds.sort(key=lambda n: n.rd)
                    n = seeds.pop(0)
                    # mark n as processed
                    # find n's neighbors
                    self._processed(n)
                    n_neighbors = self._neighbors(n)
                    # if p has a core_distance...
                    if self._core_distance(n, n_neighbors) is not None:
                        # update reachability_distance for each of n's neighbors
                        self._update(n_neighbors, n, seeds)
        # when all points have been processed
        # return the ordered list
        return self.ordered

    def cluster(self, cluster_threshold):
        clusters = []
        separators = []
        for i in range(len(self.ordered)):
            this_i = i
            next_i = i + 1
            this_p = self.ordered[i]
            if this_p.rd != None:
                this_rd = this_p.rd
            else:
                this_rd = float('infinity')
            # use an upper limit to separate the clusters
            if this_rd > cluster_threshold:
                separators.append(this_i)
        separators.append(len(self.ordered))

        for i in range(len(separators) - 1):
            start = separators[i]
            end = separators[i + 1]
            if end - start >= self.min_cluster_size:
                clusters.append(Cluster(self.ordered[start:end]))
        return clusters


class Cluster:
    def __init__(self, points):
        self.points = points

    # --------------------------------------------------------------------------
    # calculate the centroid for the cluster
    # --------------------------------------------------------------------------

    def centroid(self):
        center = [sum([p._x_center for p in self.points]) / len(self.points),
                     sum([p._y_center for p in self.points]) / len(self.points)]
        return center

def get_color(num):
    return np.random.randint(0, 255, size=(3))


def to_rgb(bw_im):
    instances = np.unique(bw_im)
    instances = instances[instances != 0]
    rgb_im = [np.zeros(bw_im.shape), np.zeros(bw_im.shape), np.zeros(bw_im.shape)]
    for instance in instances:
        color = get_color(instance)
        rgb_im[0][instance == bw_im] = color[0]
        rgb_im[1][instance == bw_im] = color[1]
        rgb_im[2][instance == bw_im] = color[2]
    return np.concatenate([np.concatenate([np.expand_dims(rgb_im[0], -1), np.expand_dims(rgb_im[1], -1)], 2), np.expand_dims(rgb_im[2], -1)], 2)


def regress_centers(Image):
  Image = np.squeeze(Image)
  instances = np.unique(Image)
  instances = instances[instances > 1000]

  mask = np.zeros_like(Image)
  mask[np.where(Image > 1000)] = 1

  centroid_regression = np.zeros([Image.shape[0], Image.shape[1], 3])
  centroid_regression[:, :, 2] = mask

  for instance in instances:
    # step A - get a center (x,y) for each instance
    instance_pixels = np.where(Image == instance)
    y_c, x_c = np.mean(instance_pixels[0]), np.mean(instance_pixels[1])
    # step B - calculate dist_x, dist_y of each pixel of instance from its center
    y_dist = (-y_c + instance_pixels[0])
    x_dist = (-x_c + instance_pixels[1])
    for y, x, d_y, d_x in zip(instance_pixels[0], instance_pixels[1], y_dist, x_dist):
      centroid_regression[y, x, :2] = [d_y, d_x]  # remember - y is distance in rows, x in columns
  return centroid_regression


def calc_clusters_img(raw_img):
    pts_list = pre_process(raw_img)
    min_cluster_size = FLAGS.min_cluster_size
    epsilon = FLAGS.epsilon
    cluster_limmit = FLAGS.cluster_limmit
    op = Optics(pts_list, min_cluster_size, epsilon)
    order = op.run()
    clusters = op.cluster(cluster_limmit)
    new_img = np.zeros((raw_img.shape[0], raw_img.shape[1]))
    for i, cluster in zip(range(len(clusters)), clusters):
        for pt in cluster.points:
            new_img[pt._row, pt._col] = (i + 1)
    return to_rgb(new_img)

'''
img0 = Image.open('/home/alon-ran/PycharmProjects/alon_mtl_8_2_18/pro_pics/pic1.png').resize([512, 256])
img1 = Image.open('/home/alon-ran/PycharmProjects/alon_mtl_8_2_18/pro_pics/pic1.png').resize([512, 256])
img0 = np.array(img0)
img1 = np.array(img1)

img_reg0 = regress_centers(img0)
img_reg1 = regress_centers(img1)

pts_list0 = pre_process(img_reg0)
min_cluster_size = 5
epsilon = 5
op = Optics(pts_list0, min_cluster_size, epsilon)
order = op.run()
clusters = op.cluster(1)
new_img = np.zeros(img0.shape)
for i, cluster in zip(range(len(clusters)), clusters):
    for pt in cluster.points:
        new_img[pt._row, pt._col] = (i + 1)

plt.imshow(new_img)
to_rgb(new_img)
im = Image.fromarray(np.uint8(to_rgb(new_img))).resize([512, 256])
im.show()

def foo(im):
    sum = 0
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i, j] != 0:
                sum += 1
                #print(im[i, j])
    return sum


print(foo(new_img))
'''






