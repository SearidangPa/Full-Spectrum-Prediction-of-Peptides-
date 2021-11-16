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
model_PATH = "model3.pt"

model = ShallowNet()
criterion = nn.MSELoss()  #mean squared error
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, \
	betas=(0.9, 0.999), eps=1e-08, weight_decay= 0.0001, amsgrad=False)

load_model_continue_training = False
if load_model_continue_training == True:
	checkpoint = torch.load(model_PATH)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	model.train()

Files = ["ProteomeTools.mgf", "NIST_Synthetic.mgf", "NIST.mgf", "MassIVE.mgf"]
chunk_size = 100

Num_data_clean = load("../Data/Num_data_clean.npy")
num_chunks_total = np.floor_divide(Num_data_clean, chunk_size)

file_index = 0
for filename in Files:

	num_chunks_train = int(num_chunks_total[file_index]*0.75)

	#permute the dataset
	num_chunks_passed = 1 
	epoch = 10

	for t in range(epoch):
		new_chunk_loss = 0

		for chunk_index in np.random.permutation(num_chunks_train):
			old_chunk_loss = new_chunk_loss

			X_val = load("../Data/" + filename.strip(".mgf") +'/one_hot_encoding' + \
							str(chunk_index) + '.npy', allow_pickle=True)

			PATH = '../Data/'+filename.strip(".mgf")+'/bin_vs_intensity' + \
													str(chunk_index) + '.npz'
			Y_val = sparse.load_npz(PATH).toarray()
			
			#convert to torch
			X_val = torch.from_numpy(X_val).float()
			X_val = torch.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1], X_val.shape[2]))
			Y_val = torch.from_numpy(Y_val).float()

			new_chunk_loss = train_model(model, X_val, Y_val, optimizer, criterion)

			#plateau condition
			if abs(old_chunk_loss - new_chunk_loss) < 0.000001:
				torch.save({'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict()
					}, model_PATH)
				break

			if num_chunks_passed % 50 ==0: #save model every 50 chunks for backup sake 
				torch.save({'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict()
					}, model_PATH)

			num_chunks_passed += 1

		print("-----------epoch number------------:", t)

	file_index += 1





