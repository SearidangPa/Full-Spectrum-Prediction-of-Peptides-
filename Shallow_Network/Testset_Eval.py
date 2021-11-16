# first neural network with keras tutorial
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from numpy import save, load
from scipy import sparse 

class ShallowNet(nn.Module):
	def __init__(self):
		super(ShallowNet, self).__init__()

		self.conv3 = nn.Conv2d(1, 64, (21, 3), 1, padding = (0,1))
		self.conv5 = nn.Conv2d(1, 64, (21, 5), 1, padding = (0,2))
		self.conv7 = nn.Conv2d(1, 64, (21, 7), 1, padding = (0,3))
		self.conv9 = nn.Conv2d(1, 64, (21, 9), 1, padding = (0,4))


		self.conv_2d_layer_2= nn.Conv2d(1, 32, 3, stride=1, padding = (1,1))
		self.conv_2d_layer_3= nn.Conv2d(32, 32, 3, stride=1, padding = (1,1))

		self.pool = nn.MaxPool2d(2, 2)

		self.fc1 = nn.Linear(32 * 64 * 6, 1000)
		self.fc2 = nn.Linear(1000, 10000)
		self.out = nn.Linear(10000, 20000)
		

	def forward(self, x):
		x3 = self.conv3(x)
		x5 = self.conv5(x)
		x7 = self.conv7(x)
		x9 = self.conv9(x)
		m = nn.PReLU()
		

		x3 = m(x3)
		x5 = m(x5)
		x7 = m(x7)
		x9 = m(x9)


		x =  torch.cat((x3, x5, x7, x9), 1)
		x = torch.reshape(x, (x.shape[0], 1, 64*4, 25))
		
		x = self.conv_2d_layer_2(x)
		x = self.pool(m(x))

		x = self.conv_2d_layer_3(x)
		x = self.pool(m(x))

		x = torch.flatten(x, 1)
		
		x = self.fc1(x)
		x = m(x)

		x = self.fc2(x)
		x = m(x)

		x = self.out(x)
		x = torch.sigmoid(x)

		return x


#initialize the model 
model = ShallowNet()
Metric = ["cosine_similarity", "euclidean_dist"]

for metric in Metric: 
	if metric == Metric[0]:
		criterion = nn.CosineEmbeddingLoss()
		model_PATH = "model_final.pt" 	 #name of the saved model 
	else:
		criterion = nn.MSELoss()
		model_PATH = "model4.pt" 	 #name of the saved model 

	#load the model
	checkpoint = torch.load(model_PATH)
	model.load_state_dict(checkpoint['model_state_dict'])

	chunk_size = 100
	num_data_clean_test = int(load("Data/num_data_clean_test.npy")[0])
	num_chunks_total = np.floor_divide(num_data_clean_test, chunk_size)


	# -----------------Evaluation ------------------
	model.eval()
	chunk_error = 0

	filename = "hcd_testingset.mgf"

	#load the test set chunk by chunk 
	for chunk_index in range(num_chunks_total):
		X_val = load("Data/"+filename.strip(".mgf") +'/one_hot_encoding' + \
						str(chunk_index) + '.npy', allow_pickle=True)

		PATH = "Data/" + filename.strip(".mgf")+'/bin_vs_intensity' + \
						str(chunk_index) + '.npz'
		
		Y_val = sparse.load_npz(PATH).toarray()
		
		#convert to torch
		X_val = torch.from_numpy(X_val)
		X_val = torch.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1], X_val.shape[2]))

		Y_val = torch.from_numpy(Y_val)

		#prediction
		Y_pred = model(X_val.float())

		#calculate the error 
		if metric == Metric[0]: 			#cosine similarity 
			chunk_error += criterion(Y_pred, Y_val, \
				torch.ones(X_val.shape[0]) ).detach().numpy()
		else:
			chunk_error += criterion(Y_pred, Y_val).detach().numpy()

		print(chunk_error)

	#divide sum_squared_error by the number of chunks in the cross validation set 
	#to get mean_squared_error
	mean_error = chunk_error  / num_chunks_total

	print(metric + " mean_error: ", mean_error)
	save(metric + "mean_error_test_set: ", mean_error)





