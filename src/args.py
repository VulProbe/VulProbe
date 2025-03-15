from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProgramArguments:
    do_train: bool = field(default=False, metadata={'help': 'Train the LineProbe.'})
    # do_test: bool = field(default=False, metadata={'help': 'Test the LineProbe.'})
    customize: bool = field(default=False, metadata={'help': 'Customize the AST.'})
    # do_visualization: bool = field(default=False, metadata={'help': 'Run visualizations.'})
    graph: bool = field(default=False, metadata={'help': 'Graph the results.'})
    
    lang: Optional[str] = field(
        default='c',
        metadata={'help': 'Language to be used in this experiment.'}
    )
    
    first_step_model_type: Optional[str] = field(
        default='LineVul',
        metadata={'help': 'Type of the first step model.'}
    )
    
    first_step_model: Optional[str] = field(
        default='/model_c.bin',
        metadata={'help': 'File name of the first_step_model.'}
    )
    
    probe_saved_path: Optional[str] = field(
        default='./results/probes',
        metadata={'help': 'Path where to save the trained probes.'}
    )
    
    probe_name: Optional[str] = field(
        default=None,
        metadata={'help': 'Name to identify the running and logging directory of LineProbe.'}
    )
    
    dataset_path: Optional[str] = field(
        default='./src/resource/dataset',
        metadata={'help': 'Path to the folder that contains the dataset.'}
    )
    
    layer: Optional[int] = field(
        default=5,
        metadata={'help': 'Layer used to get the embeddings.'}
    )
    
    strategy: Optional[str] = field(
        default='plain',
        metadata={'help': 'Strategy of LineProbe.'}
    )
    
    top_k: Optional[int] = field(
        default=5,
        metadata={'help': 'Number of top-k lines to be considered.'}
    )
    explain_method: Optional[str] = field(
        default='attention'
    )
    
    # Training Parameter
    lr: float = field(
        default=1e-3,
        metadata={'help': 'The initial learning rate for AdamW.'}
    )

    epochs: Optional[int] = field(
        default=20,
        metadata={'help': 'Number of training epochs.'}
    )

    batch_size: Optional[int] = field(
        default=32,
        metadata={'help': 'Train and validation batch size.'}
    )

    patience: Optional[int] = field(
        default=5,
        metadata={'help': 'Patience for early stopping.'}
    )
    
    rank: Optional[int] = field(
        default=128,
        metadata={'help': 'Maximum rank of the probe.'}
    )

    orthogonal_reg: float = field(
        default=5,
        metadata={'help': 'Orthogonal regularized term.'}
    )

    hidden: Optional[int] = field(
        default=768,
        metadata={'help': 'Dimension of the feature word vectors.'}
    )

    seed: Optional[int] = field(
        default=42,
        metadata={'help': 'Seed for experiments replication.'}
    )

    max_tokens: Optional[int] = field(
        default=512,
        metadata={'help': 'Max tokens considered.'}
    )
    
    name: Optional[str] = field(
        default='123456',
        metadata={'help': 'Name of the experiment.'}
    )