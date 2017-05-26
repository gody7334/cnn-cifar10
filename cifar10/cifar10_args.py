import sys, argparse

'''
parser = argparse.ArgumentParser(description='train cnn args.')
parser.add_argument('-f')
parser.add_argument('-train_dir', help="Directory where to write event logs and checkpoint.", default="/tmp/cifar10_train")
parser.add_argument('-data_dir', help="Path to the CIFAR-10 data directory.", default="/tmp/cifar10_data")
parser.add_argument('-max_steps', help="Number of batches to run.", default=10000000,type=int)
parser.add_argument('-batch_size', help="Number of images to process in a batch.", default=128,type=int)
parser.add_argument('-log_device_placement', help="Whether to log device placement.", default=False,type=bool)
parser.add_argument('-use_fp16', help="Train the model using fp16.", default=False,type=bool)
parser.add_argument('-log_frequency', help="How often to log results to the console.", default=10,type=int)

parser.add_argument('-eval_dir', help="Directory where to write event logs.", default="/tmp/cifar10_eval")
parser.add_argument('-eval_data', help="Either 'test' or 'train_eval'.", default="test")
parser.add_argument('-checkpoint_dir', help="Directory where to read model checkpoints.", default="/tmp/cifar10_train")
parser.add_argument('-eval_interval_secs', help="How often to run the eval.", default=60,type=int)
parser.add_argument('-num_examples', help="Number of examples to run.", default=1000,type=int)
parser.add_argument('-run_once', help="Whether to run eval only once.", default=False,type=bool)

args = parser.parse_args()
'''

class Arguments(object):
    train_dir = "/home/ipython/cnn-cifar10/tb_log/default/train"
    data_dir = "/home/ipython/cnn-cifar10/data"
    max_steps = 10000000
    batch_size = 256
    log_device_placement = False
    use_fp16 = False
    log_frequency = 10
    eval_dir = "/home/ipython/cnn-cifar10/tb_log/default/eval"
    eval_data = "test"
    checkpoint_dir = "/home/ipython/cnn-cifar10/tb_log/default/train"
    eval_interval_secs = 60
    num_examples = 10000
    run_once = False

    '''
    def __init__(self,
                 train_dir = "/home/ipython/cnn-cifar10/tb_log/default/train",
                 data_dir = "/home/ipython/cnn-cifar10/data",
                 max_steps = 10000000,
                 batch_size = 128,
                 log_device_placement = False,
                 use_fp16 = False,
                 log_frequency = 10,
                 eval_dir = "/home/ipython/cnn-cifar10/tb_log/default/eval",
                 eval_data = "test",
                 checkpoint_dir = "/home/ipython/cnn-cifar10/tb_log/default/train",
                 eval_interval_secs = 60,
                 num_examples = 10000,
                 run_once = False):
        
        self.train_dir = train_dir
        self.data_dir = data_dir
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.log_device_placement = log_device_placement
        self.use_fp16 = use_fp16
        self.log_frequency = log_frequency
        self.eval_dir = eval_dir
        self.eval_data = eval_data
        self.checkpoint_dir = checkpoint_dir
        self.eval_interval_secs = eval_interval_secs
        self.num_examples = num_examples
        self.run_once = run_once
    '''
        
    @staticmethod    
    def set_model_folder(f_str):
        Arguments.train_dir = "/home/ipython/cnn-cifar10/tb_log/"+f_str+"/train"
        Arguments.eval_dir = "/home/ipython/cnn-cifar10/tb_log/"+f_str+"/eval"
        Arguments.checkpoint_dir = "/home/ipython/cnn-cifar10/tb_log/"+f_str+"/train"
        
