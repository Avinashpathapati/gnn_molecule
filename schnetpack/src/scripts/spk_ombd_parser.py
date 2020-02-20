from optparse import OptionParser

def build_parser():
	parser = OptionParser()
	parser.add_option("-c", "--cuda", action='store_false')

	parser.add_option("-o", "--mode",
				  action="store", type="string", dest="mode")

	parser.add_option("-p", "--datapath",
				  action="store", type="string", dest="datapath")

	parser.add_option("-m", "--model_path",
				  action="store", type="string", dest="model_path")

	parser.add_option("-r", "--property",
				  action="store", type="string", dest="property")
	parser.add_option("-d", "--dataset",
				  action="store", type="string", dest="dataset")

	parser.add_option("-n", "--n_epochs",
				  action="store", type=int, dest="n_epochs")
	parser.add_option("-f", "--features",
				  action="store", type=int, dest="features", default=128)

	parser.add_option("-l", "--lr",
				  action="store", type=float, dest="lr", default=0.001)


	# parser.add_option("-r", "--data_reg",
	# 			  action="store", type="string", dest="data_reg")

	return parser


