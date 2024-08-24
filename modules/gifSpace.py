import torch
import torchvision 
import plothelper 
import autoencoder 
import variationalautoencoder 

def main():  
	data = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./datasets', transform=torchvision.transforms.ToTensor(), download=True), batch_size=128, shuffle=True) 
	x, y = data.__iter__().next() # hack to grab a batch
	x1 = x[y == 1][1].to(device) # find a 1
	x2 = x[y == 0][1].to(device) # find a 0
	autoenc = autoencoder.Autoencoder(2) 
	autoencoder.load(autoenc) 
	
	varautoenc = variationalautoencoder.VariationalAutoencoder(2) 
	variationalautoencoder.load(varautoenc) 
	
	plotHelper = plothelper.PlotHelper() 
	plotHelper.interpolate(autoenc, x1, x2, n=20) 
	plotHelper.interpolate(varautoenc, x1, x2, n=20) 
	plotHelper.interpolate_gif(varautoenc, x1, x2)
	
if __name__ == "__main__": 
	main() 