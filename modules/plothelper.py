import autoencoder 
from torchsummary import summary
import torchvision
import numpy as np
import torch
import matplotlib.pyplot as plt
import time 

plt.rcParams['figure.dpi'] = 200

device = "cuda" if torch.cuda.is_available() else "cpu"

class PlotHelper: 
	def __init__(self): 
		pass 
		
	# Look at the 2D latent space. [[x1, y1], [x2, y2],...]
	def plot_latent(self, autoencoder:autoencoder.Autoencoder, data:torch.utils.data.DataLoader, number_of_batches:int=64)->None:
		assert number_of_batches > 1
		for i , (x, y) in enumerate(data): 
			z = autoencoder.encoder(x.to(device))
			z = z.to(device).cpu().detach().numpy()
			plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
			if i > number_of_batches: 
				plt.colorbar() 
				break
		manager = plt.get_current_fig_manager()
		manager.window.showMaximized()
		plt.savefig("test.png")
		plt.show() 	

	# generic latent spaces
	def plot_latent_exp(self, autoencoder:autoencoder.Autoencoder, data:torch.utils.data.DataLoader, number_of_batches:int=64)->None:
		assert number_of_batches > 1
		for i , x in enumerate(data):
			z = autoencoder.encoder(x.to(device))
			z = z.to(device).cpu().detach().numpy()
			plt.scatter(z[:, 0], z[:, 1], cmap='tab10')
			if i > number_of_batches:
				plt.colorbar()
				break
		manager = plt.get_current_fig_manager()
		#manager.window.showMaximized()
		plt.savefig("test.png")
		plt.show()


	def plot_reconstructed(self, autoencoder:autoencoder.Autoencoder, r0=(-5, 10), r1=(-10, 5), n:int=12)->None: 
		width=28
		image = np.zeros((n*width, n*width))
		# 12 equidistant values between -10, 5. 		
		for j, y in enumerate(np.linspace(*r1, n)): 
			#print("loop " + str(j))
			for i, x in enumerate(np.linspace(*r0, n)): 
				z=torch.Tensor([[x,y]]).to(device)
				xHat = autoencoder.decoder(z) 
				# Reshape and convert to ndarray array
				xHat = xHat.reshape(28, 28).to('cpu').detach().numpy() 
				image[(n-1-j)*width:(n-1-j +1)*width, i*width:(i +1)*width] = xHat 
		
		manager = plt.get_current_fig_manager()
		manager.window.showMaximized()
		plt.imshow(image, extent=[*r0, *r1])
		plt.show() 
		
	def print_data_loader(self, data:torch.utils.data.DataLoader)->None: 
		for i , (x, y) in enumerate(data): 
			print(i) 
			print((x, y))
	
	def interpolate(self, autoencoder:autoencoder.Autoencoder, x1, x2, n:int=12): 
		assert n > 1 
		z_1 = autoencoder.encoder(x1)
		z_2 = autoencoder.encoder(x2)
		z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
		interpolateList = autoencoder.decoder(z)
		interpolateList = interpolateList.to('cpu').detach().numpy()

		w = 28
		img = np.zeros((w, n*w))
		for i, x_hat in enumerate(interpolateList):
			img[:, i*w:(i+1)*w] = x_hat.reshape(28, 28)
		plt.imshow(img)
		plt.xticks([])
		plt.yticks([]) 
	
	def interpolate_gif(autoencoder, filename:str, x1:int, x2:int, n:int=100):
		z1 = autoencoder.encoder(x1)
		z2 = autoencoder.encoder(x2)

		z = torch.stack([z1 + (z2 - z1)*t for t in np.linspace(0, 1, n)])

		interpolateList = autoencoder.decoder(z)
		interpolateList = interpolateList.to('cpu').detach().numpy()*255

		imagesList = [Image.fromarray(img.reshape(28, 28)).resize((256, 256)) for img in interpolateList]
		imagesList = imagesList + imagesList[::-1] # loop back beginning

		imagesList[0].save(
			f'{filename}.gif',
			save_all=True,
			append_images=imagesList[1:],
			loop=1)
