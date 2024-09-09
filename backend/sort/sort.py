"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
import hashlib
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    try:
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)
        
        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
            + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
        return o
    except Exception as e:
        print(f"Error in iou_batch: {e}")
        return np.zeros((bb_test.shape[0], bb_gt.shape[0]))  # return a zero matrix in case of error


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    # Define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = self.generate_unique_id(bbox)
    print(f"Generated unique ID: {self.id}")
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def generate_unique_id(self, bbox):
    """
    Generates a unique hash ID based on the current time and bounding box coordinates.
    """
    hash_input = f"{time.time()}_{bbox}".encode('utf-8')
    return hashlib.md5(hash_input).hexdigest()

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]

        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            try:
                pos = self.trackers[t].predict()[0]
                trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
                if np.any(np.isnan(pos)):
                    to_del.append(t)
            except Exception as e:
                print(f"Error predicting tracker {t}: {e}")
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        try:
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)
        except Exception as e:
            print(f"Error in association: {e}")
            matched, unmatched_dets, unmatched_trks = np.empty((0, 2), dtype=int), np.arange(len(dets)), np.empty((0, 5), dtype=int)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                try:
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    trk.update(dets[d, :][0])
                except Exception as e:
                    print(f"Error updating tracker {t}: {e}")

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            try:
                trk = KalmanBoxTracker(dets[i, :])
                self.trackers.append(trk)
            except Exception as e:
                print(f"Error creating new tracker for detection {i}: {e}")
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))



def parse_args():
  """
  Parse input arguments.
  """
  parser = argparse.ArgumentParser(description='SORT demo')
  parser.add_argument('--seq_path', type=str, default='data', help='Path to detections')
  parser.add_argument('--phase', type=str, default='train', help='Subdirectory in seq_path')
  parser.add_argument('--max_age', type=int, default=3, help='Maximum number of frames to keep alive a track without associated detections.')
  parser.add_argument('--min_hits', type=int, default=3, help='Minimum number of associated detections before track is initialised.')
  parser.add_argument('--iou_threshold', type=float, default=0.3, help='Minimum IOU for match.')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = parse_args()
  display = False

  if(display):
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')
    ax1.set_title('Tracked Objects')
    ax1.set_xlim(0, 1920)
    ax1.set_ylim(0, 1080)
    ax1.invert_yaxis()
    im = ax1.imshow(np.zeros((1080,1920,3),dtype=np.uint8))

  colours = np.random.rand(32,3) #used only for display
  if not os.path.exists('output'):
    os.makedirs('output')
  pattern = os.path.join(args.seq_path, args.phase, 'img1', '*.jpg')
  files = sorted(glob.glob(pattern))
  total_time = 0.0

  if(len(files) > 0):
    for frame in files:
      detfile = frame.replace('img1', 'det').replace('jpg', 'txt')
      dets = np.loadtxt(detfile, delimiter=',')
      dets = dets[:, 2:7]
      total_time += 1
      trackers = mot_tracker.update(dets)
      for d in trackers:
        d = d.astype(np.int32)
        if(display):
          im.set_data(frame)
          ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2]-d[0], d[3]-d[1], fill=False, lw=3, ec=colours[d[4]%32,:]))
          ax1.text(d[0], d[1], d[4], fontsize=15, color='white', bbox=dict(facecolor='red', alpha=0.5))
      if(display):
        fig.canvas.flush_events()
        plt.show()
  else:
    seqs = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(args.seq_path, args.phase, '*')))]
    for seq in seqs:
      print("Processing %s." % seq)
      pattern = os.path.join(args.seq_path, args.phase, seq, 'img1', '*.jpg')
      files = sorted(glob.glob(pattern))
      if(len(files) == 0):
        print("No files found.")
        continue

      total_time = 0.0
      mot_tracker = Sort(args.max_age, args.min_hits, args.iou_threshold) #create instance of the SORT tracker
      with open(os.path.join('output', '%s.txt' % seq), 'w') as out_file:
        for frame in files:
          detfile = frame.replace('img1', 'det').replace('jpg', 'txt')
          dets = np.loadtxt(detfile, delimiter=',')
          dets = dets[:, 2:7]
          total_time += 1
          trackers = mot_tracker.update(dets)
          for d in trackers:
            d = d.astype(np.int32)
            out_file.write('%d,%d,%d,%d,%d,%d\n' % (total_time, d[4], d[0], d[1], d[2]-d[0], d[3]-d[1]))
  print("Total time: %.2f" % total_time)
