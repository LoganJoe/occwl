import numpy as np

from occwl.geometry.interval import Interval


class Box:
    """A 2d or 3d box for points defined as numpy arrays"""

    def __init__(self, pt=None):
        self.intervals = []
        if pt is not None:
            for value in pt:
                self.intervals.append(Interval(value, value))

    def encompass_box(self, box):
        if len(self.intervals) == 0:
            for interval in box.intervals:
                self.intervals.append(interval)
        else:
            assert len(self.intervals) == len(box.intervals)
            for i, interval in enumerate(box.intervals):
                self.intervals[i].encompass_interval(interval)

    def encompass_point(self, point):
        if len(self.intervals) == 0:
            for i, value in enumerate(point):
                self.intervals.append(Interval(value, value))
        else:
            assert len(self.intervals) == point.size
            for i, value in enumerate(point):
                self.intervals[i].encompass_value(value)

    def contains_point(self, point):
        assert len(self.intervals) == point.size
        for i, value in enumerate(point):
            if not self.intervals[i].contains_value(value):
                return False
        return True

    def contains_box(self, box):
        assert len(self.intervals) == len(box.intervals)
        for i, interval in enumerate(self.intervals):
            if not interval.contains_interval(box.intervals[i]):
                return False
        return True

    def x_length(self):
        assert len(self.intervals) >= 1
        return self.intervals[0].length()

    def y_length(self):
        assert len(self.intervals) >= 2
        return self.intervals[1].length()

    def z_length(self):
        assert len(self.intervals) >= 2
        return self.intervals[2].length()

    def max_box_length(self):
        max_length = 0.0
        for interval in self.intervals:
            length = interval.length()
            if length > max_length:
                max_length = length
        return max_length

    def min_point(self):
        return np.array([interval.a for interval in self.intervals])

    def max_point(self):
        return np.array([interval.b for interval in self.intervals])

    def diagonal(self):
        return self.max_point() - self.min_point()

    def center(self):
        return np.array([interval.middle() for interval in self.intervals])

    def offset(self, dist):
        for i in range(len(self.intervals)):
            self.intervals[i].offset(dist)

    def get_bounding_box(self):
        if len(self.intervals) < 3:
            raise ValueError("Box must be 3D to get bounding box data")

        min_pt = self.min_point()
        max_pt = self.max_point()

        xmin, ymin, zmin = min_pt[0], min_pt[1], min_pt[2]
        xmax, ymax, zmax = max_pt[0], max_pt[1], max_pt[2]

        return [
            "%.2f" % xmin, "%.2f" % ymin, "%.2f" % zmin,
            "%.2f" % xmax, "%.2f" % ymax, "%.2f" % zmax,
            "%.2f" % (xmax - xmin), "%.2f" % (ymax - ymin), "%.2f" % (zmax - zmin)
        ]
