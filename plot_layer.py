import torch 
import autoencoder
from autoencoder import Decoder
import logger

def main(): 
	logging = logger.BLSCLogger() 
	log = logging.setup_standard_logger("aelogger", "log.txt")
	autoenc = autoencoder.Autoencoder(2, log) 
	autoencoder.load(autoenc, "model_weights.pth") 
	print(autoenc.decoder.__getattr__(autoenc.decoder.linear2.name).weight)

if __name__ == "__main__": 
	main() 