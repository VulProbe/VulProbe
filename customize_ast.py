import os
from datetime import datetime

def main():
    first_step_model_type = 'codebert'
    # ['codebert', 'mlp', 'bilstm', 'unixcoder']:
    # name = '00'
    for name in ['00', '11', '22', '33', '44']:
    # for name in ['00']:
        first_step_model = f'model_{name}.bin'
        probe_model = f'{first_step_model_type}_c_128_model_{name}_probe_42'
        device = '0'
        batch_size = 1
        command = f"CUDA_VISIBLE_DEVICES={device} python src/main.py --customize --probe_name {probe_model} " \
                f"--first_step_model {first_step_model}  " \
                f"--first_step_model_type {first_step_model_type} " \
                f"--batch_size {batch_size} " \
                f"--hidden 64"
                
        print(command)
        os.system(command)

if __name__ == '__main__':
    main()
