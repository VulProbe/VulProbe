import os
from datetime import datetime

def main():
    now = datetime.now()
    month_day = now.strftime("%m%d")
    
    # CodeBERT
    for lang in ['c']:
        model_seed = '00'
        first_step_model = f"model_{model_seed}.bin"
        first_step_model_type = "codebert"
        device = '3'
        seed = '42'
        probe_name = '_'.join(['codebert', lang, '128', 'model', model_seed, 'probe', seed])
        print(f"CUDA_VISIBLE_DEVICES={device} python src/main.py --do_train --probe_name {probe_name} "
                    f"--lang {lang} --first_step_model {first_step_model} "
                    f" --first_step_model_type {first_step_model_type} --rank 128 --seed {seed}")
        os.system(f"CUDA_VISIBLE_DEVICES={device} python src/main.py --do_train --probe_name {probe_name} "
                    f"--lang {lang} --first_step_model {first_step_model} "
                    f" --first_step_model_type {first_step_model_type} --rank 128 --seed {seed}")
        

if __name__ == '__main__':
    main()
