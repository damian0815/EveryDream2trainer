import os
import yaml
import argparse

def count_multiplied_set_helper(root_dir, log_depth):
    if not os.path.isdir(root_dir):
        return 0
    count = 0
    multiply = 1
    for filename in os.listdir(root_dir):
        full_path = os.path.join(root_dir, filename)
        if filename == 'global.yaml':
            # load yaml and read 'multiply' value
            with open(full_path, "r") as f:
                config = yaml.safe_load(f)
                multiply = config.get('multiply', multiply)
                #print(f"Found multiply in {filename}: {multiply}")
        elif os.path.isdir(full_path):
            count += count_multiplied_set_helper(full_path, log_depth-1)
        elif filename.endswith(".txt"):
            count += 1

    if log_depth >= 0:
        print(count*multiply, multiply, root_dir)
    return count * multiply



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("root_dir")
    arg_parser.add_argument("--max_log_depth", type=int, default=0, help="Max depth to log counts for")

    args = arg_parser.parse_args()

    count_multiplied_set_helper(args.root_dir, args.max_log_depth)
    #print(count)
