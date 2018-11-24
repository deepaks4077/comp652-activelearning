import cPickle
with open('cifar-10-batches-py/data_batch_1', 'rb') as fo:
	dict = cPickle.load(fo)
    	print dict

