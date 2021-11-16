# first neural network with keras tutorial
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from numpy import save, load
from scipy import sparse 

class ShallowNet1(nn.Module):
	def __init__(self):
		super(ShallowNet1, self).__init__()

		self.conv3 = nn.Conv2d(1, 64, (21, 3), 1, padding = (0,1))
		self.conv5 = nn.Conv2d(1, 64, (21, 5), 1, padding = (0,2))
		self.conv7 = nn.Conv2d(1, 64, (21, 7), 1, padding = (0,3))
		self.conv9 = nn.Conv2d(1, 64, (21, 9), 1, padding = (0,4))


		self.conv_2d_layer_2= nn.Conv2d(1, 32, 3, stride=1, padding = (1,1))
		self.conv_2d_layer_3= nn.Conv2d(32, 32, 3, stride=1, padding = (1,1))

		self.fc1 = nn.Linear(32 * 256 * 25, 2000)
		self.out = nn.Linear(2000, 20000)

	def forward(self, x):
		x3 = self.conv3(x)
		x5 = self.conv5(x)
		x7 = self.conv7(x)
		x9 = self.conv9(x)

		x3 = F.relu(x3)
		x5 = F.relu(x5)
		x7 = F.relu(x7)
		x9 = F.relu(x9)

		x =  torch.cat((x3, x5, x7, x9), 1)
		x = torch.reshape(x, (x.shape[0], 1, 64*4, 25))
		
		x = self.conv_2d_layer_2(x)
		x = F.relu(x)

		x = self.conv_2d_layer_3(x)
		x = F.relu(x)

		x = torch.flatten(x, 1)
		
		x = self.fc1(x)
		x = F.relu(x)

		x = self.out(x)
		x = torch.sigmoid(x)

		return x

class ShallowNet2(nn.Module):
	def __init__(self):
		super(ShallowNet2, self).__init__()

		self.conv3 = nn.Conv2d(1, 64, (21, 3), 1, padding = (0,1))
		self.conv5 = nn.Conv2d(1, 64, (21, 5), 1, padding = (0,2))
		self.conv7 = nn.Conv2d(1, 64, (21, 7), 1, padding = (0,3))
		self.conv9 = nn.Conv2d(1, 64, (21, 9), 1, padding = (0,4))


		self.conv_2d_layer_2= nn.Conv2d(1, 32, 3, stride=1, padding = (1,1))
		self.conv_2d_layer_3= nn.Conv2d(32, 32, 3, stride=1, padding = (1,1))

		self.fc1 = nn.Linear(32 * 256 * 25, 1000)
		self.fc2 = nn.Linear(1000, 1000)
		self.out = nn.Linear(1000, 20000)

	def forward(self, x):
		x3 = self.conv3(x)
		x5 = self.conv5(x)
		x7 = self.conv7(x)
		x9 = self.conv9(x)

		x3 = F.relu(x3)
		x5 = F.relu(x5)
		x7 = F.relu(x7)
		x9 = F.relu(x9)

		x =  torch.cat((x3, x5, x7, x9), 1)
		x = torch.reshape(x, (x.shape[0], 1, 64*4, 25))
		
		x = self.conv_2d_layer_2(x)
		x = F.relu(x)

		x = self.conv_2d_layer_3(x)
		x = F.relu(x)

		x = torch.flatten(x, 1)
		
		x = self.fc1(x)
		x = F.relu(x)

		x = self.fc2(x)
		x = F.relu(x)

		x = self.out(x)
		x = torch.sigmoid(x)

		return x

class ShallowNet3(nn.Module):
	def __init__(self):
		super(ShallowNet3, self).__init__()

		self.conv3 = nn.Conv2d(1, 64, (21, 3), 1, padding = (0,1))
		self.conv5 = nn.Conv2d(1, 64, (21, 5), 1, padding = (0,2))
		self.conv7 = nn.Conv2d(1, 64, (21, 7), 1, padding = (0,3))
		self.conv9 = nn.Conv2d(1, 64, (21, 9), 1, padding = (0,4))


		self.conv_2d_layer_2= nn.Conv2d(1, 32, 3, stride=1, padding = (1,1))
		self.conv_2d_layer_3= nn.Conv2d(32, 32, 3, stride=1, padding = (1,1))

		self.pool = nn.MaxPool2d(2, 2)

		self.fc1 = nn.Linear(32 * 64 * 6, 10000)
		self.out = nn.Linear(10000, 20000)

	def forward(self, x):
		x3 = self.conv3(x)
		x5 = self.conv5(x)
		x7 = self.conv7(x)
		x9 = self.conv9(x)

		x3 = F.relu(x3)
		x5 = F.relu(x5)
		x7 = F.relu(x7)
		x9 = F.relu(x9)

		x =  torch.cat((x3, x5, x7, x9), 1)
		x = torch.reshape(x, (x.shape[0], 1, 64*4, 25))
		
		x = self.conv_2d_layer_2(x)
		x = self.pool(F.relu(x))

		x = self.conv_2d_layer_3(x)
		x = self.pool(F.relu(x))

		x = torch.flatten(x, 1)
		
		x = self.fc1(x)
		x = F.relu(x)

		x = self.out(x)
		x = torch.sigmoid(x)

		return x

class ShallowNet4(nn.Module):
	def __init__(self):
		super(ShallowNet4, self).__init__()

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

class ShallowNet5(nn.Module):
	def __init__(self):
		super(ShallowNet5, self).__init__()

		self.conv3 = nn.Conv2d(1, 64, (21, 3), 1, padding = (0,1))
		self.conv5 = nn.Conv2d(1, 64, (21, 5), 1, padding = (0,2))
		self.conv7 = nn.Conv2d(1, 64, (21, 7), 1, padding = (0,3))
		self.conv9 = nn.Conv2d(1, 64, (21, 9), 1, padding = (0,4))


		self.conv_2d_layer_2= nn.Conv2d(1, 32, 3, stride=1, padding = (1,1))
		self.conv_2d_layer_3= nn.Conv2d(32, 32, 3, stride=1, padding = (1,1))

		self.pool = nn.MaxPool2d(2, 2)

		self.fc1 = nn.Linear(32 * 64 * 6, 1000)
		self.fc2 = nn.Linear(1000, 4000)
		self.fc3 = nn.Linear(4000, 10000)
		self.out = nn.Linear(10000, 20000)
		

	def forward(self, x):
		x3 = self.conv3(x)
		x5 = self.conv5(x)
		x7 = self.conv7(x)
		x9 = self.conv9(x)

		
		x3 = F.relu(x3)
		x5 = F.relu(x5)
		x7 = F.relu(x7)
		x9 = F.relu(x9)


		x =  torch.cat((x3, x5, x7, x9), 1)
		x = torch.reshape(x, (x.shape[0], 1, 64*4, 25))

		x = self.conv_2d_layer_2(x)
		x = self.pool(F.relu(x))

		x = self.conv_2d_layer_3(x)
		x = self.pool(F.relu(x))

		x = torch.flatten(x, 1)
		
		x = self.fc1(x)
		x = F.relu(x)

		x = self.fc2(x)
		x = F.relu(x)

		x = self.fc3(x)
		x = F.relu(x)

		x = self.out(x)
		x = torch.sigmoid(x)

		return x

def train_model(model, X_train, y_train, optimizer, criterion):
	batch_size = 10;
	#reference to stackoverflow
	#https://stackoverflow.com/questions/45113245/how-to-get-mini-batches-in-pytorch-in-a-clean-and-efficient-way
	permutation = torch.randperm(X_train.size()[0])
	chunk_loss = 0 
	for i in range(0, X_train.size()[0], batch_size):
		optimizer.zero_grad()

		indices = permutation[i:i+batch_size]
		batch_x, batch_y = X_train[indices], y_train[indices]

		outputs = model(batch_x)

		loss = criterion(outputs, batch_y)
		loss.backward()
		optimizer.step()

		chunk_loss += loss.item()

		print("loss batch:", loss.item())

	print("--------chunk_loss-------: ", chunk_loss)
	return chunk_loss	



#initialize the model 
model= ["model1.pt", "model2.pt", "model3.pt", \
		"model4.pt", "model5.pt"]

model_PATH = model[4] #change manually

if model_PATH == model[0]:
	model = ShallowNet1()
elif model_PATH == model[1]:
	model = ShallowNet2()
elif model_PATH == model[2]:
	model = ShallowNet3()
elif model_PATH == model[3]:
	model = ShallowNet4()
elif model_PATH == model[4]:
	model = ShallowNet5()


checkpoint = torch.load(model_PATH)
model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.MSELoss()  #mean squared error
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, \
	betas=(0.9, 0.999), eps=1e-08, weight_decay= 0, amsgrad=False)


Files = ["ProteomeTools.mgf", "NIST_Synthetic.mgf", "NIST.mgf", "MassIVE.mgf"]
chunk_size = 100

Num_data_clean = load("../Data/Num_data_clean.npy")
num_chunks_total = np.floor_divide(Num_data_clean, chunk_size)


# -----------------Evaluation ------------------
model.eval()
mean_batch_squared_error = 0
mean_squared_error = 0

#load cross_validation set 
file_index = 0

for filename in Files:
	num_chunks_train = int(num_chunks_total[file_index]*0.75)

	#the chunks not trained is the last 25%
	for chunk_index in range(num_chunks_train, int(num_chunks_total[file_index,0])):
		X_val = load("../Data/" + filename.strip(".mgf") +'/one_hot_encoding' + \
						str(chunk_index) + '.npy', allow_pickle=True)

		PATH = '../Data/'+filename.strip(".mgf")+'/bin_vs_intensity' + \
						str(chunk_index) + '.npz'
		
		Y_val = sparse.load_npz(PATH).toarray()
		
		#convert to torch
		X_val = torch.from_numpy(X_val)
		X_val = torch.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1], X_val.shape[2]))

		Y_val = torch.from_numpy(Y_val)

		#prediction
		Y_pred = model(X_val.float())

		#calculate sum of squared error
		mean_batch_squared_error += criterion(Y_pred, Y_val).detach().numpy()

	#divide sum_squared_error by the number of chunks in the cross validation set 
	#to get mean_squared_error for each file 
	mean_squared_error += mean_batch_squared_error  / (num_chunks_total[file_index] - num_chunks_train)
	file_index += 1

mean_squared_error = mean_squared_error / 4 #divide by four because we have four files
print("mean_squared_error: ", mean_squared_error)





