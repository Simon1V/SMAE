from torch.utils.data import DataLoader, Dataset 
import numpy as np 
from torchvision import datasets 
import torchvision.transforms as T 
import os 
from PIL import Image 

default_transformation = T.Compose([T.Resize(size=(28,28)), T.ToTensor()])

class DSHelper: 
	def __init__(self, training_data, verification_data=None, labeled:bool=False):
		if labeled == True:  
			self.train_features, self.train_labels = next(iter(training_data))
		else: 
			self.train_features = next(iter(training_data))
			
	def get_image_data(self):
		pass 

	def get_image_by_index(self, index:int): 
		# May need to call cpu.detach() if the tensors are on the CPU during training? 
		img = self.train_features[0].numpy() 
		normalized_image = (img - np.min(img)) * (255.0 / (np.max(img) - np.min(img)))
		return normalized_image

	def get_image_by_index2(self, index:int):
		data = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./datasets', transform=torchvision.transforms.ToTensor(), download=True), batch_size=128, shuffle=True) 
		imgs_labels = [imglabel for imglabel in data]
		return imgs_labels[index][0][0][0]
		
# Assumes classes are available.
class DSCustom: 
	def __init__(self, training_directory_path:str, verification_directory_path:str): 

		data_transform = T.Compose([T.Resize(size=(26,26)), T.RandomHorizontalFlip(p=0.5), T.ToTensor()])
		self.training_data = datasets.ImageFolder(root=training_directory_path, transform=data_transform, target_transform=None)
		self.verification_data = datasets.ImageFolder(root=verification_directory_path, transform=data_transform)

	def get_data_loaders(self): 
		training_dataloader = DataLoader(dataset=self.training_data, batch_size=1, num_workers=1, shuffle=True) 
		verification_dataloader = DataLoader(dataset=self.verification_data, batch_size=1, num_workers=1, shuffle=True)
		return training_dataloader, verification_dataloader


# General custom dataset. 
class DSCustomGeneral(Dataset): 
	def __init__(self, target_directory:str, transform=default_transformation): 
		self.target_directory = target_directory
		self.paths = [] 	
		files = os.listdir(self.target_directory)
		for file in files: 
			if self.getFileExtension(file) in ["jpeg", "jpg", "png"]: 
				self.paths.append(os.path.abspath(os.path.join( self.target_directory, file)))
			 
		self.transform = transform 

	def __len__(self): 
		return len(self.paths)

	# Here the images should be treated with float representation. 
	def __getitem__(self, index): 
		image_path = self.paths[index] 
		image = Image.open(image_path)
		image_tensor = self.transform(image)
		return image_tensor
	
	def getFileExtension(self, file:str) -> str: 
		return file.split('.')[-1].lower() 

class DSCSVGeneral(Dataset): 
	def __init__(self, target_csv_directory:str): 
		self.target_csv = target_csv_directory
		self.dataframe = pd.read_csv(self.target_csv) 	
		self.rows = [] 
		for index, row in self.dataframe.iterrows():
			rows.append(row.tolist())

	def __len__(self): 
		return len(self.rows)

	# Here the images should be treated with float representation. 
	def __getitem__(self, index):
		assert len(self.rows) >= index 
		return self.rows[index]