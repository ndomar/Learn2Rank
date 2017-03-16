#!/usr/bin/env python
"""
This module provides functionalities for parsing the data
"""

import os
import csv
import numpy as np
import datetime
from topia.termextract import extract

class DataParser(object):

	def __init__(self, dataset):
		"""
		Initializes the data parser given a dataset name
		"""
		self.base_folder = '../datasets/Extended_ctr'
		self.dataset = dataset
		
		if dataset == 'citeulike-a':
			self.dataset_folder = self.base_folder + '/citeulike_a_extended'
		elif dataset == 'citeulike-t':
			self.dataset_folder = self.base_folder + '/citeulike-t_extended'
		else:
			print("Warning: Given dataset not known, setting to citeulike-a")
			self.dataset_folder = self.base_folder + '/citeulike_a_extended'

		self.paper_count = None
		self.user_count = None

		self.process()
	

	def process(self):
		"""
		Starts parsing the data and gets matrices ready for training
		"""
		self.feature_labels, self.feature_matrix = self.parse_paper_features()
		self.raw_labels, self.raw_data = self.parse_paper_raw_data()
		self.ratings = self.generate_ratings_matrix()

	def parse_paper_features(self):
		"""
		Parses paper features
		"""
		now = datetime.datetime.now()
		path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.dataset_folder, 'paper_info.csv')
		with open(path, "r") as f:			
			reader = csv.reader(f, delimiter='\t')
			first_line = True
			feature_vec = []
			i = 0
			row_length = 0
			labels_ids = []
			for line in reader:
				if first_line:
					labels = ["type", "publisher", "year", "address"]
					for j, entry in enumerate(line):
						if entry in labels:
							labels_ids.append(j)
					row_length = len(labels_ids)
					first_line = False
					i += 1
					continue
				paper_id = line[0]
				if int(paper_id) != i:
					for _ in range(int(paper_id) - i):
						feature_vec.append([None] * row_length)
						i += 1
				current_entry = []
				for k, label_id in enumerate(labels_ids):
					if k == 2:
						current_entry.append(now.year - int(line[label_id]))
					else:
						current_entry.append(line[label_id])
				feature_vec.append(current_entry)
				i += 1
		
		if self.paper_count is None:
			self.paper_count = len(feature_vec)
		return labels, np.array(feature_vec)

	def parse_paper_raw_data(self):
		"""
		Parses paper raw data
		"""
		path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.dataset_folder, 'raw-data.csv')
		delimiter = ','
		if self.dataset == 'citeulike-t':
			delimiter = '\t'
		with open(path, "r") as f:
			reader = csv.reader(f, delimiter=delimiter)
			first_line = True
			data_vec = []
			row_length = 0
			for line in reader:
				if first_line:
					labels = line[1:]
					row_length = len(line)
					first_line = False
					continue
				data_vec.append(line[1:])

		if self.paper_count is None:
			self.paper_count = len(data_vec)

		return labels, np.array(data_vec)

	def generate_ratings_matrix(self):
		"""
		Generates a rating matrix of user x paper
		"""
		if self.paper_count is None:
			self.raw_labels, self.raw_data = self.parse_paper_raw_data()

		path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.dataset_folder, 'users.dat')
		self.user_count = sum(1 for line in open(path))
		ratings = np.zeros((self.user_count, self.paper_count))
		i = 0
		with open(path, "r") as f:
			for line in f:
				splitted = line.replace("\n", "").split(" ")
				for paper_id in splitted:
					ratings[i][int(paper_id)] = 1
				
				i += 1

		return ratings

	def get_raw_data(self):
		return self.raw_labels, self.raw_data

	def get_feature_vector(self):
		return self.feature_labels, feature_matrix

	def get_ratings_matrix(self):
		return self.ratings
				



