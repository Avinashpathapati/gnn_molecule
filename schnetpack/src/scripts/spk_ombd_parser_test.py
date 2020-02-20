from optparse import OptionParser

def build_parser():
	parser = OptionParser()

	parser.add_option("-t", "--test_path",
				  action="store", type="string", dest="test_path")


	# parser.add_option("-r", "--data_reg",
	# 			  action="store", type="string", dest="data_reg")

	return parser


