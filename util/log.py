from collections import OrderedDict
import hashlib, os


class color:
 BOLD   = '\033[1m\033[48m'
 END    = '\033[0m'
 ORANGE = '\033[38;5;202m'
 BLACK  = '\033[38;5;240m'


def create_logger(args):
  from torch.utils.tensorboard import SummaryWriter
  """Use hyperparms to set a directory to output diagnostic files."""

  arg_dict = args.__dict__
  assert "seed" in arg_dict, \
    "You must provide a 'seed' key in your command line arguments"
  assert "logdir" in arg_dict, \
    "You must provide a 'logdir' key in your command line arguments."
  assert "env_name" in arg_dict, \
    "You must provide a 'env_name' key in your command line arguments."

  # sort the keys so the same hyperparameters will always have the same hash
  arg_dict = OrderedDict(sorted(arg_dict.items(), key=lambda t: t[0]))

  # remove seed so it doesn't get hashed, store value for filename
  # same for logging directory
  seed = str(arg_dict.pop("seed"))
  logdir = str(arg_dict.pop('logdir'))
  env_name = str(arg_dict.pop('env_name'))

  # get a unique hash for the hyperparameter settings, truncated at 10 chars
  arg_hash   = hashlib.md5(str(arg_dict).encode('ascii')).hexdigest()[0:6] + '-seed' + seed
  logdir     = os.path.join(logdir, env_name)
  output_dir = os.path.join(logdir, arg_hash)

  # create a directory with the hyperparm hash as its name, if it doesn't
  # already exist.
  os.makedirs(output_dir, exist_ok=True)

  # Create a file with all the hyperparam settings in plaintext
  info_path = os.path.join(output_dir, "experiment.info")
  file = open(info_path, 'w')
  for key, val in arg_dict.items():
      file.write("%s: %s" % (key, val))
      file.write('\n')

  logger = SummaryWriter(output_dir, flush_secs=0.1)
  print("Logging to " + color.BOLD + color.ORANGE + str(output_dir) + color.END)

  logger.dir = output_dir
  return logger


