import logging
import sys
import os
import yaml
import imp
import pprint

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.DEBUG,
    stream=sys.stdout,
)
def deploy(yaml_filepath):
      cfg = load_cfg(yaml_filepath)

      # Print the configuration - just to make sure that you loaded what you
      # wanted to load
      pp = pprint.PrettyPrinter(indent=4)
      pp.pprint(cfg)
      print("cfg", cfg)

      # Here is an example how you load modules of which you put the path in the
      # configuration. Use this for configuring the model you use, for dataset
      # loading, ...
      dpath = cfg["deploy"]["script_path"]
      #os.system("cd ..")
      sys.path.insert(1, os.path.dirname(dpath))
      #ex = "python3 "+dpath+" -d "+cfg['dataset']['processed_path']+" -r "+cfg['dataset']['raw_files_path']

      #print(ex)
      os.system("python3 "
                +dpath)
    #data = imp.load_source("data", cfg["dataset"]["script_path"])
    #pp.pprint(data)
def run_ingest(yaml_filepath):
      cfg = load_cfg(yaml_filepath)

      # Print the configuration - just to make sure that you loaded what you
      # wanted to load
      pp = pprint.PrettyPrinter(indent=4)
      pp.pprint(cfg)
      print("cfg", cfg)

      # Here is an example how you load modules of which you put the path in the
      # configuration. Use this for configuring the model you use, for dataset
      # loading, ...
      dpath = cfg["dataset"]["script_path"]
      #os.system("cd ..")
      sys.path.insert(1, os.path.dirname(dpath))
      #ex = "python3 "+dpath+" -d "+cfg['dataset']['processed_path']+" -r "+cfg['dataset']['raw_files_path']

      #print(ex)
      os.system("python3 "
                +dpath
                +" -d "+cfg['dataset']['processed_dir']
                +" -r "+cfg['dataset']['rawfiles_dir'])
    #data = imp.load_source("data", cfg["dataset"]["script_path"])
    #pp.pprint(data)
def run_pre_process(yaml_filepath):
      cfg = load_cfg(yaml_filepath)

      # Print the configuration - just to make sure that you loaded what you
      # wanted to load
      pp = pprint.PrettyPrinter(indent=4)
      pp.pprint(cfg)
      print("cfg", cfg)

      # Here is an example how you load modules of which you put the path in the
      # configuration. Use this for configuring the model you use, for dataset
      # loading, ...
      dpath = cfg["pre_process"]["script_path"]
      #os.system("cd ..")
      sys.path.insert(1, os.path.dirname(dpath))
      #ex = "python3 "+dpath+" -d "+cfg['dataset']['processed_path']+" -r "+cfg['dataset']['raw_files_path']

      #print(ex)
      os.system("python3 "
                +dpath
                +" -m "+cfg['pre_process']['model_dir']
                +" -df "+cfg['pre_process']['storage_dir'])
    #data = imp.load_source("data", cfg["dataset"]["script_path"])
    #pp.pprint(data)
def run_training(yaml_filepath):
      cfg = load_cfg(yaml_filepath)

      # Print the configuration - just to make sure that you loaded what you
      # wanted to load
      pp = pprint.PrettyPrinter(indent=4)
      pp.pprint(cfg)
      print("cfg", cfg)

      # Here is an example how you load modules of which you put the path in the
      # configuration. Use this for configuring the model you use, for dataset
      # loading, ...
      dpath = cfg["training"]["script_path"]
      #os.system("cd ..")
      sys.path.insert(1, os.path.dirname(dpath))
      #ex = "python3 "+dpath+" -d "+cfg['dataset']['processed_path']+" -r "+cfg['dataset']['raw_files_path']

      #print(ex)
      os.system("python3 "
                +dpath
                +" -d "+cfg['training']['model_dir']
                +" -df "+cfg['training']['preprocess_dir']
                +" -c "+cfg['training']['config_path'])
    #data = imp.load_source("data", cfg["dataset"]["script_path"])
    #pp.pprint(data)
def main(args):
    """Example."""

    if args.ingest:
      run_ingest(args.filename)
    if args.preprocess:
      run_pre_process(args.filename)
    if args.train:
      run_training(args.filename)
    if args.deploy:
      deploy(args.filename)
      


def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    cfg : dict
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, "r") as stream:
        cfg = yaml.load(stream)
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    return cfg


def make_paths_absolute(dir_, cfg):
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Parameters
    ----------
    dir_ : str
    cfg : dict

    Returns
    -------
    cfg : dict
    """
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.isfile(cfg[key]):
                logging.error("%s does not exist.", cfg[key])

        if key.endswith("_dir"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
        
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg

def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t", True, "True", "TRUE"]:
        return True
    elif value in ["false", "no", "n", "0", "f", False, "False", "FALSE"]:
        return False

    return True


def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="filename",
        help="experiment definition file",
        metavar="FILE",
        required=True,
    )
    parser.add_argument("--ingest", 
                        dest="ingest",
                        type = parse_boolean, 
                        default = True, 
                        help = "set this flag as true for ingesting data from raw files")
    parser.add_argument("--preprocess", 
                        dest="preprocess",
                        type = parse_boolean, 
                        default = True, 
                        help = "set this flag as true for pre-processing the data")

    parser.add_argument("--train", 
                        dest="train",
                        type = parse_boolean, 
                        default = True, 
                        help = "set this flag as true for pre-processing the data")
    parser.add_argument("--deploy", 
                        dest="deploy",
                        type = parse_boolean, 
                        default = True, 
                        help = "set this flag as true for pre-processing the data")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    print(args)
    main(args)