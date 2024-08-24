import autoencoder 
from torchsummary import summary
import torchvision
import numpy as np
import torch
import modules.plothelper as plothelper
import datasethelper 
import logger 
import argparse
import os 
import re 
import sys 
from PIL import Image 
import torchvision.transforms as T 
import matplotlib.pyplot as plt 

device = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_BASE = "/media/simon/Environment/Software Projects/Autoencoders/Self Modelling Autoencoder/datasets/Custom/"

description={"L": "latent representation", "B" : "batch inference", "S" :"single inference"}

def main(): 
	logging = logger.BLSCLogger() 
	log = logging.setup_standard_logger("aelogger", "log.txt")
	log.info("Initialization")
	parser = argparse.ArgumentParser()
	parser.add_argument("-lD", "--latent-dimension", help="Latent dimension", action="store",type=int , default=2)

	subparsers = parser.add_subparsers(dest="command")

	# Inference specific 
	parser_inference = subparsers.add_parser("inference", help="Run inference")
	parser_inference.add_argument("-w", "--weights", type=str,  help="model weights path", action="store", default="model_weights.pth") 
	parser_inference.add_argument("-m", "--mode", type=str, choices="SBL", help="S for Single Inference, B for batch inference, L for latent space visualisatiion",  default="S")
	parser_inference.add_argument("-t", "--target", type=str, help="Single inference target", action="store", default="test.png")
	parser_inference.add_argument("-d", "--datapath", help="Dataset Base Path for latent representation", action="store", default=DEFAULT_BASE)
	parser_inference.add_argument("-dS", "--datasetname", help="Dataset name for latent representation",type=str, action="store")


	# Training specific 
	parser_training = subparsers.add_parser("training", help="Run training")
	parser_training.add_argument("-w", "--weights", type=str,  help="model weights path", action="store", default="model_weights.pth") 
	parser_training.add_argument("-d", "--datapath", help="Dataset Base Path", action="store", default=DEFAULT_BASE)
	parser_training.add_argument("-dS", "--datasetname", help="Dataset name",type=str, action="store")
	parser_training.add_argument("-lr", "--learningrate", type=float, help="Learning rate", action="store", default=0.01)
	parser_training.add_argument("-b", "--batch-size", type=int, help="Batch size", action="store")
	parser_training.add_argument("-e", "--epochs", type=int, help="Number of epochs", action="store", default=10) 

	args_main = parser.parse_args()
	if args_main.command=="training": 
		log.info("Run training")
		args_training, _ = parser_training.parse_known_args()
		autoenc = autoencoder.Autoencoder(args_main.latent_dimension, log) 
		autoenc.to(device)
		if args_training.datasetname == "mnist": 
			log.info("Training on mnist default ds")
			data = torch.utils.data.DataLoader(torchvision.datasets.MNIST('/home/datasets/mnist', transform=torchvision.transforms.ToTensor(), download=True, train=True), batch_size=128, shuffle=True) 
			model = autoencoder.train(autoenc, data, epochs=20 , lr=args_training.learningrate)
			autoencoder.save(model)
		else: 
			log.info("Training on custom dataset.")
			ds_absolute_path = os.path.join(args_training.datapath, args_training.datasetname)
			log.debug("Datasetpath %s", ds_absolute_path)
			training_dataset = datasethelper.DSCustomGeneral(ds_absolute_path)
			
			#pretransformed data, no need to set a transformation.  
			training_data = torch.utils.data.DataLoader(training_dataset,batch_size=128, num_workers=2, shuffle=True )
			model = autoencoder.train(autoenc, training_data, epochs=args_training.epochs , lr=args_training.learningrate)
			autoencoder.save(model)
			
	elif args_main.command=="inference": 
		log.info("Run inference") 
		args_inference, _ = parser_inference.parse_known_args()
		log.debug("Weights: " + args_inference.weights)
		log.debug("Running in " + description[args_inference.mode] + " mode")
		autoenc = autoencoder.Autoencoder(args_main.latent_dimension, log) 
		autoencoder.load(autoenc, args_inference.weights) 
		if args_inference.mode == "S": 
			autoencoder.load(autoenc, args_inference.weights)
			image = Image.open(args_inference.target)
			image = image.resize((28,28)) 
			image.save("testScaled.png")
			
			if image.mode == "L": 
				image = image.convert("RGB")

			converter = T.ToTensor()
			image_tensor = converter(image)
			image_tensor.to(device)
			image_tensor_decoded = autoenc(image_tensor) 
			image_tensor_decoded = torch.reshape(image_tensor_decoded, (3, 28, 28))
			restored_image = T.ToPILImage()(image_tensor_decoded)
			plt.imshow(restored_image)
			plt.show()	

		elif args_inference.mode == "B": 
			plthelper = plothelper.PlotHelper()
			plthelper.plot_reconstructed(autoenc)  
		
		# make it possible to plot non mnist data. 	
		elif args_inference.mode == "L":
			plthelper = plothelper.PlotHelper() 
			#training_data = torch.utils.data.DataLoader(torchvision.datasets.MNIST('/home/datasets', transform=torchvision.transforms.ToTensor(), download=True), batch_size=128, shuffle=True)
			ds_relative_path = os.path.join(args_inference.datapath, args_inference.datasetname)
			training_dataset = datasethelper.DSCustomGeneral(ds_relative_path)
			training_data = torch.utils.data.DataLoader(training_dataset,batch_size=64, num_workers=2, shuffle=True )

			plthelper.plot_latent_exp(autoenc, training_data)
		
		else: 
			log.error("Invalid mode specified.")
			sys.exit(-1)

	else: 
		log.warning("No run mode specified, aborting.")
		sys.exit(-1) 

if __name__ == "__main__": 
	main() 			
