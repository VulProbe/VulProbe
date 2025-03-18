import os
import logging
logger = logging.getLogger(__name__)

# VulProbe(CodeBERT): model_type = 'codebert', explain_method = 'probe'
# VulProbe(UnixCoder): model_type = 'unixcoder', explain_method = 'probe'
# VulProbe(bilstm): model_type = 'bilstm', explain_method = 'probe'
# VulProbe(mlp): model_type = 'mlp', explain_method = 'probe'

# strategy = 'plain': tree+uni
# strategy = 'ast': tree+uni+preorder

def main():
    model_type = 'codebert' 
    # ['bilstm', 'mlp', 'codebert', 'unixcoder', 'LineVul', 'linevd', 'qwen', 'gpt']
    # for name in ['00', '11', '22', '33', '44']:
    for name in ['11']:
        strategy = 'plain' 
        # ['plain', 'ast']
        for explain_method in ['saliency']:
        # for explain_method in ['saliency', 'deeplift_shap', 'gradient_shap']:
        # for explain_method in ['qwen', 'gpt, 'linevul', 'probe', 'linevd']: 
            for top_k in [1, 3, 5, 10]:
            # for top_k in [1]:
                command = f"python src/expalaination_evaluation.py " \
                    f"--explain_method {explain_method} " \
                    f"--top_k {top_k} " \
                    f"--first_step_model_type {model_type} " \
                    f"--strategy {strategy} " \
                    f"--name {name}"
                print(command)
                os.system(command)

if __name__ == '__main__':
    main()
