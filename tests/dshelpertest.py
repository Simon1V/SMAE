import datasethelper
import torch 
import torchvision
import matplotlib.pyplot as plt 
import os 

def test1(): 
	data = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./datasets', transform=torchvision.transforms.ToTensor(), download=True), batch_size=128, shuffle=True) 
	ds_helper = datasethelper.DSHelper(data)
	image = ds_helper.get_image_by_index(12)
	image = image[0]
	plt.imshow(image)
	plt.show() 
	
def test2():
	dataset = datasethelper.DSCustomGeneral("datasets/Custom/sequences")
	#pretransformed data, no need to set a transformation.  
	dataloader = torch.utils.data.DataLoader(dataset,batch_size=1, num_workers=2, shuffle=True )
	ds_helper  = datasethelper.DSHelper(dataloader)
	image = ds_helper.get_image_by_index(2)
	image = image[0]
	plt.imshow(image)
	plt.show() 

def main(): 
	test2()
	
if __name__ == "__main__": 
	main() 
