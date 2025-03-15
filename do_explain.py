import os
import logging
logger = logging.getLogger(__name__)

def main():
    
    # for name in ['00', '11', '22', '33', '44']:
    for name in ['11', '22', '33', '44']:
        for lang in ['c']:
            # for model_type in ['codebert', 'unixcoder', 'mlp', 'bilstm', 'codet5', 'graphcodebert', 'unixcoder']:
            for model_type in ['codebert']:
                # for strategy in ['plain', frequency', 'ast']:
                for strategy in ['ast']:
                    print(f"python src/do_probe_explain.py --first_step_model_type {model_type} "
                            f"--lang {lang} --strategy {strategy} --name {name}")
                    os.system(f"python src/do_probe_explain.py --first_step_model_type {model_type} "
                            f"--lang {lang} --strategy {strategy} --name {name}")


if __name__ == '__main__':
    main()
