import numpy as np
from random import *

def get_mrc_dimensions(file_to_open):
  x_min = y_min = z_min= float("inf")
  x_max = y_max = z_max = -1
  with open(file_to_open) as input_file:
    for line in input_file:
      x_coord, y_coord, z_coord, thresh, label = line.strip().split(",")
      x_coord = int(x_coord)
      y_coord = int(y_coord)
      z_coord = int(z_coord)
      if x_coord < x_min:
        x_min = x_coord
      if y_coord < y_min:
        y_min = y_coord
      if z_coord < z_min:
        z_min = z_coord
      if x_coord > x_max:
        x_max = x_coord
      if y_coord > y_max:
        y_max = y_coord
      if z_coord > z_max:
        z_max = z_coord
  return x_max, x_min, y_max, y_min, z_max, z_min

def writePredictionsToFile(predicted_labels, num_lines, index, axis_x, axis_y, axis_z):
  output_path = "./" + str(index) + ".txt"
  file = open(output_path,"w+")
  for i in range(0, num_lines):
    file.write(str(axis_x[0][i]) + " " + str(axis_y[0][i]) + " " + str(axis_z[0][i]) + " " + str(predicted_labels[i]) + "\r\n")

def print_batch_stats(train_accuracy, current_loss):
  print('training accuracy %g' % (train_accuracy))
  print('Loss %g' % (current_loss))

def select_patch_random(a, b, x_length, y_length, z_length, patch_width, patch_height, patch_depth):
  x_start = randint(0, x_length - patch_width)
  y_start = randint(0, y_length - patch_height)
  z_start = randint(0, z_length - patch_depth)
  a = np.delete(a, slice(0, x_start), axis=1)
  a = np.delete(a, slice(patch_width, x_length - x_start), axis=1)
  a = np.delete(a, slice(0, y_start), axis=2)
  a = np.delete(a, slice(patch_height, y_length - y_start), axis=2)
  a = np.delete(a, slice(0, z_start), axis=3)
  a = np.delete(a, slice(patch_depth, z_length - z_start), axis=3)
  b = np.delete(b, slice(0, x_start), axis=1)
  b = np.delete(b, slice(patch_width, x_length - x_start), axis=1)
  b = np.delete(b, slice(0, y_start), axis=2)
  b = np.delete(b, slice(patch_height, y_length - y_start), axis=2)
  b = np.delete(b, slice(0, z_start), axis=3)
  b = np.delete(b, slice(patch_depth, z_length - z_start), axis=3)
  b = np.reshape(b, (-1, patch_width * patch_height * patch_depth, 3))
  return a, b

def read_mrc_image_data(file_path, patch_width, patch_height, patch_depth, batch_size, file_num):
  x_max, x_min, y_max, y_min, z_max, z_min = get_mrc_dimensions(file_path)
  x_length = x_max - x_min + 1
  y_length = y_max - y_min + 1
  z_length = z_max - z_min + 1
  with open(file_path) as f:
    a = np.zeros((batch_size, x_length, y_length, z_length, 1), dtype = float)
    b = np.zeros((batch_size, x_length, y_length, z_length, 3), dtype = float)
    next(f)
    for line in f:
      x_coord, y_coord, z_coord, thresh, label = line.strip().split(",")
      x_pos = int(x_coord) - x_min
      y_pos = int(y_coord) - y_min
      z_pos = int(z_coord) - z_min
      thresh = float(thresh)
      label = int(label)
      batch_it = file_num % batch_size
      a[batch_it][x_pos][y_pos][z_pos][0] = thresh
      b[batch_it][x_pos][y_pos][z_pos][label] = 1
    a, b = select_patch_random(a, b, x_length, y_length, z_length, patch_width, patch_height, patch_depth)
  return a, b


