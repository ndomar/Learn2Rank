#!/usr/bin/env python
"""
This module provides functionalities for extracting peer papers
"""
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from util.top_similar import TopSimilar
import sys

class PeerExtractor(object):


	def __init__(self, ratings, documents, method, similarity_metric, k):
		
		self.ratings = ratings
		self.method = method
		self.k = k
		self.documents = documents
		self.similarity_metric = similarity_metric
		self.pairs = {}
		self.similarity_matrix = None
		self.calculate_pairwise_similarity()

	def get_similarity_matrix(self):
		return self.similarity_matrix

	def get_user_peer_papers(self, user):

		if self.method == 'random':
			self.pairs[user] = self.get_random_peer_papers(user)
		else:
			if self.method == 'least_k':
				self.pairs[user] = self.get_least_k(user)
			else: 
				self.pairs[user] = get_least_similar_k(user)

		return self.pairs[user]

	def get_least_k(self, user):
		poitive_papers = self.ratings[user].nonzero()[0]
		negative_papers = np.where(self.ratings[user] == 0)[0]
		user_ratings = self.ratings[user]
		top_similar = TopSimilar(self.k)
		for index, rating in enumerate(user_ratings):
			top_similar.insert(index, 1 - rating)
		return top_similar.get_indices()

	def get_least_similar_k(self, user):
		poitive_papers = self.ratings[user].nonzero()[0]
		negative_papers = np.where(self.ratings[user] == 0)[0]
		user_ratings = self.ratings[user].nonzero()[0]
		top_similar = TopSimilar(self.k)
		for index in user_ratings:
			rating = 1 - self.ratings[user][index]
			top_similar.insert(index, rating)
		return top_similar.get_indices()		


	def get_random_peer_papers(self, user):

		if user in self.pairs:
			return self.pairs[user]

		positive_papers = self.ratings[user].nonzero()[0]
		negative_papers = np.where(self.ratings[user] == 0)[0]
		pairs = []
		for paper in positive_papers:
			random_indices = random.sample(range(0, len(negative_papers)), self.k)
			for index in random_indices:
				pairs.append((paper, negative_papers[index]))
		return pairs

	def calculate_pairwise_similarity(self):
		if self.similarity_matrix is not None:
			return
		docs_count = self.ratings.shape[1]
		self.similarity_matrix = np.eye(docs_count)
		if self.similarity_metric == 'dot':
			# Compute pairwise dot product
			for i in range(docs_count):
				for j in range(i, docs_count):
					self.similarity_matrix[i][j] = self.documents[i].dot(self.documents[j].T)
					self.similarity_matrix[j][i] = self.similarity_matrix[i][j]

		if self.similarity_metric == 'cosine':
			self.similarity_matrix = cosine_similarity(sparse.csr_matrix(self.documents))


