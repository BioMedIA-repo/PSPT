import time


def add_common_arguments(parser):

    # dataset related parameters
    parser.add_argument("--dataset-root", type=str, default="",
                    help="Path to strip files (unused when --scatter-png-dir is set)")
    parser.add_argument("dataset_csv", type=str, help='The csv of dataset.')
    parser.add_argument("--output-dir", type=str, help="An output directory")

    parser.add_argument("--val_fold", type=int, default=0, help="[CV] The value in fold column of CSV as validation set (default 0: fold=0=val, fold=1=train, fold<0=test)")

    parser.add_argument("--batch-size-train", type=int, default=128, help="Choose the batch size for training the backbone network")
    parser.add_argument("--batch-size-eval", type=int, default=1024, help="Choose the batch size for evaluating the backbone network")

    parser.add_argument("--dataset-name", type=str, default=None, help="bracs, coad-msi, or luad-survival")
    parser.add_argument('--data-norm', action="store_true", help='Whether to normlize data using data-mean, data-std')
    parser.add_argument('--data-mean', type=lambda s: [float(item) for item in s.split(',')], default=None,
                        help='mean of the dataset')
    parser.add_argument('--data-std', type=lambda s: [float(item) for item in s.split(',')], default=None,
                        help='std of the dataset')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of workers to use for training data loading')
    parser.add_argument('--num-workers-eval', type=int, default=0,
                        help='Number of workers to use for eval data loading')

    # model and network realted parameters
    parser.add_argument("--model", type=str, default=None, help="type of MIL model, eg. clam_mb")
    parser.add_argument("--transfer-type", type=str, default=None, help="type of transfer learning, eg. prompt, spe, adapter, lora, end2end")
    parser.add_argument("--network", type=str, default=None, help="type of backbone network, eg. uni_v1")

    parser.add_argument("--num-prompt-tokens", type=int, default=1, help="number of prompt tokens")
    parser.add_argument(
        '--deep-prompt', action="store_true",
        help='Enable deep prompts; PSPT always enables this option.'
    )
    parser.add_argument("--prompt-dropout", type=float, default=0., help="")
    parser.add_argument("--project-prompt-dim", type=int, default=-1, help="")

    # Learning rate schedule parameters
    parser.add_argument("--epochs", type=int, default=40, help="How many epochs to train for")
    parser.add_argument("--precision", type=str, default='32', help="32, 16-mixed, or bf16-mixed")
    parser.add_argument("--amp-init-scale", type=float, default=512., help="Initial FP16 gradient scale")
    parser.add_argument('--weight-decay', type=float, default=1e-2,
                        help='Weight decay of the optimizer (default: 1e-2)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr-factor', type=float, default=1.,
                        help='learning rate multiplication for pretrained networks (default: 1.)')
    parser.add_argument('--adam', action="store_true",
                        help='Use Adam optimizer if set to true, otherwise use AdamW.')
    parser.add_argument('--loss-weight', type=lambda s: [float(item) for item in s.split(',')], default=None,
                        help='Weight of each class')
    parser.add_argument('--auto-loss-weight', action="store_true", help='Automatically calculate the weight of each class')
    parser.add_argument('--accumulate-grad-batches', type=int, default=1, help='simulate larger batch size by accumulating gradients')

    # Dropout
    parser.add_argument('--dropout-inst', type=float, default=0.0, help='Dropout rate for patches')
    parser.add_argument(
        '--backbone-image-size', type=int, default=256,
        help='Spatial input size passed to the image backbone (PLIP should use 224).',
    )
    parser.add_argument('--dropout-att', type=float, default=0., help='Dropout rate for attentions')

    # pretrained weights related
    parser.add_argument('--pretrained', action="store_true", help='load imagenet pretrained weight')
    parser.add_argument('--load-backbone-weight', type=str, default=None, help='If not None, load weights from given path')
    parser.add_argument('--load-weights', type=str, default=None, help='If not None, load weights from given path')

    # gpu
    parser.add_argument('--gpu-id', type=lambda s: [int(item) for item in s.split(',')], default=None)

    # project name and tag
    parser.add_argument('--run-name', type=str, default='test')
    parser.add_argument('--tag', type=str, default='', help="For logging only")
    parser.add_argument('--weighted_sample', action="store_true", help='Enable weighted sampling for imbalanced classes')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducible evaluation')

    # ===== Adapter arguments =====
    parser.add_argument("--adapter-dim", type=int, default=64, help="Bottleneck dimension for Adapter")
    # ===== LoRA arguments =====
    parser.add_argument("--lora-r", type=int, default=8, help="Rank of LoRA")
    parser.add_argument("--lora-alpha", type=int, default=16, help="Alpha scaling for LoRA")
    # ===== SCPM: Slide-Aware Cross-Layer Prompt Modulation =====
    parser.add_argument("--scpm-latent-dim", type=int, default=128,
                        help="Latent dimension of sampled-WSI-initialized SCPM")
    parser.add_argument("--disable-scpm", action="store_true",
                        help="Ablation: replace SCPM-modulated prompts with static VPT-Deep prompts")
    # ===== FPRD: Fidelity-Preserving Residual Diffusion =====
    parser.add_argument("--enable-fprd", action="store_true",
                        help="Enable FPRD before the MIL head")
    parser.add_argument("--fprd-neighbors", type=int, default=16,
                        help="Number of FPRD feature-graph neighbors per patch")
    parser.add_argument("--fprd-temperature", type=float, default=0.2,
                        help="Softmax temperature for FPRD graph weights")
    parser.add_argument("--fprd-time-max", type=float, default=0.5,
                        help="Maximum learnable diffusion time for FPRD")
    parser.add_argument("--fprd-iterations", type=int, default=3,
                        help="Number of fixed-point iterations for FPRD")

    # ===== PCPS-selected patch loading =====
    parser.add_argument("--scatter-png-dir", type=str, default=None,
                        help="Directory containing indexed patch PNG files")
    parser.add_argument("--pcps-selection-path", type=str, default=None,
                        help="Path to an PCPS selection JSON file")
    parser.add_argument("--pcps-selected-count", type=int, default=384,
                        help="Number of task-relevant PCPS patches per training WSI")
    parser.add_argument("--pcps-random-count", type=int, default=128,
                        help="Number of random context patches per training WSI")
    parser.add_argument(
        "--pcps-sampling-mode", choices=["legacy", "evidence_context", "gumbel_top_m", "uniform"],
        default="gumbel_top_m",
        help="legacy uses selected+random counts; evidence_context uses a total bag size and evidence concentration",
    )
    parser.add_argument("--pcps-total-count", type=int, default=None,
                        help="Total training patches per WSI in evidence_context mode")
    parser.add_argument("--pcps-evidence-concentration", type=float, default=0.5,
                        help="Fraction of the training bag reserved for the deterministic high-evidence core")
    parser.add_argument("--pcps-context-temperature", type=float, default=float("inf"),
                        help="Temperature for score-guided context sampling; inf gives uniform context and exact legacy equivalence")
    parser.add_argument("--pcps-distribution-concentration", type=float, default=1.0,
                        help="Nonnegative evidence concentration kappa for unified all-patch Gumbel-Top-M sampling; 0 is uniform")
    parser.add_argument(
        "--pcps-eval-mode", choices=["full", "topk", "matched"], default="full",
        help="Evaluation bag policy: full WSI, deterministic PCPS Top-K, or a reproducible draw matching training",
    )
    parser.add_argument(
        "--pcps-eval-sampling-seed", type=int, default=None,
        help="Local validation/test sampling seed; defaults to --seed",
    )

    return parser


def get_arguments(parser):
    parser = add_common_arguments(parser)
    opts = parser.parse_args()
    opts = process_common_arguments(opts)
    return opts


def process_common_arguments(opts):
    # SCPM is defined on layer-wise prompts, so PSPT is always VPT-Deep.
    # Set the generic flag explicitly to keep saved parameters truthful.
    if getattr(opts, 'transfer_type', None) == 'pspt':
        opts.deep_prompt = True
    return opts

def get_arguments_additional(parser, add_argument_fun, process_argument_fun):
    parser = add_common_arguments(parser)
    if add_argument_fun is not None:
        parser = add_argument_fun(parser)
    opts = parser.parse_args()
    opts = process_common_arguments(opts)
    if process_argument_fun is not None:
        opts = process_argument_fun(opts)
    return opts
