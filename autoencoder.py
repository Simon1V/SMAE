import torch; torch.manual_seed(0) 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.utils 
import torch.distributions 
import torchvision 
import numpy as np 
import pickle 
import time 
import logging 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class Encoder(nn.Module): 
	def __init__(self, latentDims:int): 
		assert latentDims > 1 
		super(Encoder, self).__init__() 
		self.linear1 = nn.Linear(784, 512)
		self.linear2 = nn.Linear(512, latentDims)
		
	def forward(self, x:torch.Tensor) -> torch.Tensor: 
		x=torch.flatten(x, start_dim=1) 
		x = self.linear1(x) 
		x = F.relu(x)
		x = self.linear2(x)
		return x


class Decoder(nn.Module): 
	def __init__(self, latent_dimensions:int): 
		assert latent_dimensions > 1 
		super(Decoder, self).__init__() 
		# Restore back/ upsample information. 
		self.linear1 = nn.Linear(latent_dimensions, 512) 
		self.linear2 = nn.Linear(512, 784) 
		self.linear2.name = "selfmodellinglayer"
		
	def forward(self, z:torch.Tensor) -> torch.Tensor: 
		# Test with GELU in both layers. 
		z = F.relu(self.linear1(z)) 
		z = torch.sigmoid(self.linear2(z)) 
		self_modelling_layer = self.linear1
		return z.reshape((-1, 1, 28, 28)), self_modelling_layer 


class Autoencoder(nn.Module): 
	def __init__(self, latent_dimensions:int, logger): 
		assert latent_dimensions > 1 
		super(Autoencoder, self).__init__() 
		self.logger = logger 
		self.encoder = Encoder(latent_dimensions) 
		self.decoder = Decoder(latent_dimensions) 
		
	def forward(self, x:torch.Tensor)->torch.Tensor: 
		z = self.encoder(x) 
		reconstructed, self_modelling_layer = self.decoder(z)
		return reconstructed, self_modelling_layer

	def self_modelling_loss(self, target_self_modelling_layer:torch.Tensor): 
		#target_self_modelling_layer is a linear layer (matrix), got to convert to tensor
		target_self_modelling_layer_tensor = target_self_modelling_layer.weight.view(-1).cpu().detach().numpy() 
		median_deviation = np.median(target_self_modelling_layer_tensor)
		print(median_deviation)	
		return median_deviation  

	# Add "punishment" for weight distributions to 
	def self_modelling_loss_2(self, target_self_modelling_layer:torch.Tensor): 
		target_self_modelling_layer_tensor = target_self_modelling_layer.weight.view(-1).cpu().detach().numpy() 
		

# Validation loss missing. 	
def train(autoencoder:Autoencoder, data:torch.utils.data.DataLoader, epochs:int=20, lr:float = 0.01, labeled:bool=False, self_modelling:bool=False): 
	autoencoder.logger.info("Training with learning rate: %s ", lr)
	optimizer = torch.optim.Adam(autoencoder.parameters(), lr) 
	criterion = nn.MSELoss()
	autoencoder.logger.debug("Dataset labeled: %s", labeled)
	autoencoder.logger.debug("Self Modelling: %s", self_modelling)

	epoch_loss = 0.0 
	batch_loss = 0.0 
	
	if labeled == False:
		for epoch in range(0, epochs): 
			autoencoder.logger.info("Current Epoch: %s", epoch) 	
			for x in data: 
				x = x.to(device)
				#print(x.size(0))	 
				optimizer.zero_grad() 
				xHat, target_self_modelling_layer = autoencoder(x) 
				if self_modelling == True: 
					loss = criterion(x, xHat) + autoencoder.self_modelling_loss(target_self_modelling_layer)
				else:
					loss = criterion(x, xHat)
				loss.backward() 
				optimizer.step() 
				batch_loss = batch_loss + loss.item()
				print("Item loss: %.3f", loss.item() )
				#  running_loss += loss.item() * inputs.size(0)
				# but the input doesn't have a shape which includes the batches as first arg, just single imgs, so no use here. 
			print("Batch Loss: %.5f" % (batch_loss / len(data)))
			
			batch_loss = 0.0 
		return autoencoder 
			
	else: 
		for epoch in range(0, epochs): 
			autoencoder.logger.info("Current Epoch: %s", epoch)		
			for x, _ in data: 
				x = x.to(device)
				optimizer.zero_grad() 
				xHat, target_self_modelling_layer = autoencoder(x)
				if self_modelling == True: 
					loss = criterion(x, xHat) + autoencoder.self_modelling_loss(target_self_modelling_layer)
				else: 
					loss = criterion(x, xHat)
				loss.backward() 
				optimizer.step() 
				batch_loss = batch_loss + loss.item()
			print("Batch Loss: %.5f" % (batch_loss / len(data)))
			batch_loss = 0.0 
		return autoencoder


def save(model, name:str="model_weights.pth")->bool: 
	torch.save(model.state_dict(), name) 
	return True 


def load(model, name:str="model_weights.pth")->bool: 
	if device == "cpu": 
		model.load_state_dict(torch.load(name, map_location=device, weights_only=True))
		return True 
	elif device == "cuda": 
		model.load_state_dict(torch.load(name, weights_only=True))
		model.to(device)
		return True 
	else: 
		return False 
	
