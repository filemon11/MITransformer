import torch

import optuna
import argparse
# Note that using this function is probably not
# very safe but okay for low risk applications
# as this

from . import argtypes
from ...utils import logmaker
from ...utils.params import Undefined
from ... import data
from ...readingtimes.lme import parse as lmeparse

logger = logmaker.getLogger(__name__)
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

torch.autograd.set_detect_anomaly(False)


# TODO: for hyperopt spaces, specify whether we have a continuous space,
# a selection or both as part of the type

TransformerDescription = argtypes.OptNone(
    argtypes.StrToTuple(
        argtypes.StrToTuple(argtypes.StrToTuple(str, ...), int), ...))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rank', '--local-rank', type=argtypes.OptNone(int), default=None,
        help="which rank this process runs on; set by torchrun.")
    parser.add_argument(
        '--n_workers', type=int, default=0,
        help="number of workers for the dataloader")
    parser.add_argument(
        '--name', type=str,
        default=logmaker.get_timestr(),
        help="experiment name. Defaults to current time")
    parser.add_argument(
        '--device', type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to run models on; must be 'cuda' if --use_ddp is set")
    parser.add_argument(
        '--use_ddp', type=argtypes.str_to_bool,
        default=torch.cuda.device_count() > 1,
        help="whether to use distributed GPU training")
    parser.add_argument(
        '--use_amp', type=argtypes.str_to_bool,
        default=True,
        help="whether to use automatic mixed precision (half-precision FP16)")
    parser.add_argument(
        '--use_8bit', type=argtypes.str_to_bool,
        default=True,
        help="whether to use 8bit optimisers (including StableEmbeddings)")
    parser.add_argument(
        '--seed', type=int, default=1895,
        help="seed for random processes")
    # TODO: actually set seed

    # Subparsers
    # # Training Parser
    subparsers = parser.add_subparsers(dest="mode")
    train_parser = subparsers.add_parser(
        "train", help="training mode")
    train_parser.add_argument(
        '--n_runs', type=int,
        default=1,
        help="Number of runs; if > 1 computes mean and std of final scores.")

    # Data parser group
    data_group = train_parser.add_argument_group('data')
    data_group.add_argument(
        '--dataset_name', type=str, help='name of the dataset to load',
        default='Wikitext_processed')
    data_group.add_argument(
        '--max_len_train', type=argtypes.OptNone(int), default=40,
        help='maximum number of tokens in training set')
    data_group.add_argument(
        '--max_len_eval_test', type=argtypes.OptNone(int), default=None,
        help='maximum number of tokens in eval set')
    data_group.add_argument(
        '--min_len_train', type=argtypes.OptNone(int), default=3,
        help='minimum number of tokens in training set')
    data_group.add_argument(
        '--min_len_eval_test', type=argtypes.OptNone(int), default=None,
        help='minimum number of tokens in eval set')
    data_group.add_argument(
        '--masked', type=argtypes.str_to_bool, default=True,
        help=(
            'Whether to include dependencies in dataset. Necessary for '
            'dependency parsing setting.'))
    data_group.add_argument(
        '--triangulate', type=int, default=0,
        help='TODO')
    data_group.add_argument(
        '--vocab_size', type=argtypes.OptNone(int), default=1_000_000,
        help=(
            'number of most frequent tokens to embed; all other '
            'tokens are replaced with an UNK token;'
            'can be None when loading existing token_mapper'))
    data_group.add_argument(
        '--first_k', type=argtypes.OptNone(int), default=None,
        help='only load first k sentences of the training set')
    data_group.add_argument(
        '--first_k_eval_test', type=argtypes.OptNone(int), default=None,
        help='only load first k sentences of the eval and test sets')
    data_group.add_argument(
        '--connect_with_dummy', type=argtypes.str_to_bool, default=True,
        help=(
            'Establish an arc to a dummy token when there is no '
            'parent/child among the precedents?'))
    data_group.add_argument(
        '--connect_with_self', type=argtypes.str_to_bool, default=False,
        help=(
            'Establish a recursive arc to the token itself when there '
            'is not parent/child among the precedents?'))
    data_group.add_argument(
        '--masks_setting', type=str, choices=(
            "complete", "current", "next", "both"),
        default="current",
        help=('What dependencies to assign to the current token.'))

    # # # Trainer parser group
    trainer_group = train_parser.add_argument_group('trainer')
    trainer_group.add_argument(
        '--model_name', type=argtypes.OptNone(str),
        default=None,
        help="model name. Set to experiment name if None")
    trainer_group.add_argument(
        '--dependency_mode', type=str,
        choices=("supervised", "input", "standard"),
        default="supervised",
        help="how to use dependency information")
    trainer_group.add_argument(
        '--combined_loss', type=argtypes.str_to_bool,
        default=False,
        help=(
            "whether to use combined loss for"
            "unsupervised memory cost learning"))
    trainer_group.add_argument(
        '--distr_mode', type=str,
        default="att-n", choices=("att", "att-n"),
        help="mode for calculation attention distribution for combined loss")
    trainer_group.add_argument(
        '--global_distr', type=argtypes.str_to_bool,
        default=True,
        help=(
            "Whether to compute the distribution for the combined loss"
            " globally or as an average of per-head distributions."))
    trainer_group.add_argument(
        '--include_current', type=argtypes.str_to_bool,
        default=False,
        help=(
            "Include attention to current item (diagonal) when computing"
            "the attention distribution for attention losses."))
    trainer_group.add_argument(
        '--length_weighted', type=argtypes.str_to_bool,
        default=False,
        help=(
            "Normalise attention entropy loss by length of left "
            "context (maximum entropy). The scores are mapped to an "
            "the interval [0, 1]."))
    trainer_group.add_argument(
        '--batch_size', type=int, default=32,
        help=(
            "batch size; in case of multiple GPUs it is "
            "chunked across the devices"))
    trainer_group.add_argument(
        '--use_steps', type=argtypes.str_to_bool, default=False,
        help=(
            "Where the unit of evaluation are steps (i.e. batches) "
            "instead of epochs. If true, --eval_interval "
            "refers to number of batches processed."))
    trainer_group.add_argument(
        '--max_steps', type=argtypes.OptNone(int), default=100_000,
        help=(
            "maximum number of steps (batches) to process "
            "if --use_steps is set to true"))
    trainer_group.add_argument(
        '--eval_interval', type=int,
        default=1, help="frequency to perform evaluations in")
    trainer_group.add_argument(
        '--early_stop_after', type=argtypes.OptNone(int), default=None,
        help=(
            "abort training after x evaluations without "
            "improvement on the eval loss; if None, no early stopping"))
    trainer_group.add_argument(
        '--early_stop_metric', type=argtypes.OptNone(str),
        default='perplexity',
        help=(
            "metric to chose for early stopping if "
            "--early_stop_after is not none"))
    trainer_group.add_argument(
        '--epochs', type=int, default=100,
        help="how many epochs to train for")
    trainer_group.add_argument(
        '--gradient_acc', type=argtypes.OptNone(int), default=None,
        help="If specified, only optimises after n iterations.")
    trainer_group.add_argument(
        '--learning_rate', type=float, default=1e-3,
        help="learning rate for the optimiser")
    trainer_group.add_argument(
        '--loss_alpha', type=argtypes.OptNone(float), default=0.5,
        help=(
            "loss weight for supervised learning; 1.0 is only "
            "language model training while 0.0 is only arc training"))
    trainer_group.add_argument(
        '--losses', type=argtypes.OptNone(
            argtypes.StrToDict(str, float)), default={"lm": 1},
        help=(
            "Dictionary of losses for combined loss setting "
            "and their weights. Must include 'lm'."))
    trainer_group.add_argument(
        '--k_negatives', type=argtypes.OptNone(int),
        default=None,
        help=(
            "k negatives to approximate softmax"))
    trainer_group.add_argument(
        '--arc_loss_weighted', type=argtypes.str_to_bool, default=False,
        help="Overrepresent arcs against non-arcs in arc loss calculation")
    trainer_group.add_argument(
        '--discriminative', type=argtypes.str_to_bool, default=False,
        help="Train the language model in a discriminative fashion")

    # # # Model parser group
    model_group = train_parser.add_argument_group('model')
    model_group.add_argument(
        '--transformer_description',
        type=TransformerDescription,
        default=None,
        help=(
            "Architecture of the transformer model. Tuple of layers "
            "where each layer is a tuple of a tuple of "
            "head types and a width."
            "The width is applied to every head type in the layer."
            "If provised, overrides --layer_design, --use_standard, "
            "--width, --depth, --unrestricted_before, --unrestricted_after"))
    model_group.add_argument(
        '--layer_design',
        type=argtypes.OptNone(argtypes.StrToTuple(str, ...)), default=None,
        # ("head_current", "child_current"),
        help=(
            "design of the core transformer layer; tuple of head types "
            "is overriden if --transformer_description is provided"))
    model_group.add_argument(
        '--use_standard',
        type=argtypes.str_to_bool, default=False,
        help=("whether to add an unrestricted head to the core layer(s)"))
    model_group.add_argument(
        '--width',
        type=int, default=1,
        help=(
            "width of the core transformer; "
            "is overriden if --transformer_description is provided"))
    model_group.add_argument(
        '--depth',
        type=int, default=1,
        help=(
            "depth of the core transformer; "
            "is overriden if --transformer_description is provided"))
    model_group.add_argument(
        '--unrestricted_before',
        type=int, default=0,
        help=(
            "number of unrestricted layers below the core transformer; "
            "is overriden if --transformer_description is provided"))
    model_group.add_argument(
        '--unrestricted_after',
        type=int, default=0,
        help=(
            "number of unrestricted layers above the core transformer; "
            "is overriden if --transformer_description is provided"))
    model_group.add_argument(
        '--d_ff_factor', type=int, default=4,
        help="hidden dimensionality of the feed-forward layers (*n_embd)")
    model_group.add_argument(
        '--dropout', type=argtypes.OptNone(float), default=0.3,
        help=(
            "hidden dimensionality of the feed-forward layers; "
            "can be further specified by additional dropout params"))
    model_group.add_argument(
        '--dropout_attn', type=argtypes.OptNone(float), default=0,
        help=(
            "dropout for the attention module; "
            "overridden by --dropout if set to -1"))
    model_group.add_argument(
        '--dropout_resid', type=argtypes.OptNone(float), default=-1,
        help=(
            "dropout for the residual connections; "
            "overridden by --dropout if set to -1"))
    model_group.add_argument(
        '--dropout_ff', type=argtypes.OptNone(float), default=-1,
        help=(
            "dropout for the feed-forward layers; "
            "overridden by --dropout if set to -1"))
    model_group.add_argument(
        '--dropout_embd', type=argtypes.OptNone(float), default=-1,
        help=(
            "dropout for the embedding layer; "
            "overridden by --dropout if set to -1"))
    model_group.add_argument(
        '--dropout_lstm', type=argtypes.OptNone(float), default=-1,
        help=(
            "dropout for the LSTM (if existant); "
            "overridden by --dropout if set to -1"))
    model_group.add_argument(
        '--use_lstm', type=argtypes.str_to_bool, default=True,
        help=("use LSTM layer"))
    model_group.add_argument(
        '--block_size', type=int, default=500,
        help="maximum sequence length of the model")
    model_group.add_argument(
        '--n_embd', type=int, default=500,
        help="model embedding size")
    model_group.add_argument(
        '--overlay_causal', type=argtypes.str_to_bool, default=True,
        help=(
            "whether to overlay a casual mask if providing"
            "masks as additional inputs"))
    model_group.add_argument(
        '--use_dual_fixed', type=argtypes.str_to_bool, default=False,
        help=(
            "whether to cross-fix key and query weights of "
            "the head and child attention heads. Only allowed "
            "if both are present in the description with width 1"))
    model_group.add_argument(
        '--bias', type=argtypes.str_to_bool, default=False,
        help="Whether to use bias in all of the model weights")
    model_group.add_argument(
        '--pos_enc', type=str, choices=("embedding", "sinusoidal"),
        default="embedding",
        help="What kind of positional encodings to use.")

    # # Hyperopt Parser
    hyperopt_parser = subparsers.add_parser(
        "hyperopt", help="hyperopt mode")
    hyperopt_parser.add_argument(
        '--optimise', type=str,
        default="perplexity",
        help="metric to optimise")
    hyperopt_parser.add_argument(
        '--n_warmup_steps', type=int,
        default=1,
        help=(
            "how many evaluations to wait before pruning can "
            "happen within a trial"))
    hyperopt_parser.add_argument(
        '--sampler_startup_trials', type=int,
        default=10,
        help=(
            "how many trials to run with random sampling before using TPE."))
    hyperopt_parser.add_argument(
        '--psyling_eval', type=argtypes.str_to_bool,
        default=False,
        help=(
            "use psyling eval (also possible if not optimising for loglik)."))
    hyperopt_parser.add_argument(
        '--pruner_startup_trials', type=int,
        default=5,
        help=(
            "how many trials to run before pruning can happen at all."))
    hyperopt_parser.add_argument(
        '--sampler', type=str, choices=("tpe", "random"),
        default="random",
        help=(
            "optuna sampler to use"))
    hyperopt_parser.add_argument(
        '--pruner', type=str, choices=("median", "hyperband"),
        default="median",
        help=(
            "optuna pruner to use"))
    hyperopt_parser.add_argument(
        '--n_trials', type=int,
        default=25,
        help="how many trials to run")
    hyperopt_parser.add_argument(
        '--psyling_dataset',
        type=argtypes.StrToTuple(str, ...),
        help=(
            'name of the dataset for psycholinguistic evaluation.'
            'Can be several datasets separated via comma. These are '
            'concatenated by the optimiser.'),
        default="naturalstories")
    hyperopt_parser.add_argument(
        '--average_psyling',
        type=argtypes.str_to_bool,
        default=False,
        help=(
            'If True, fits lme separately to each psyling '
            'dataset and takes the average -loglik per row '
            'as the hyperopt goal. If False, concatenates the '
            'datasets and fits only one lme. In this setting '
            'you can specify group random effects for the '
            'corpus using the term "Corpus".'),
    )
    hyperopt_parser.add_argument(
        '--load_psyling_mmap',
        type=argtypes.OptNone(str),
        default=None,
        help=(
            'If specified, loads mmap of psycholinguistic'
            ' evaluation dataset from .temp directory. This'
            ' argument specifies the file name.'),)
    hyperopt_parser.add_argument(
        '--shift', type=int, default=0,
        help=(
            'Argument for adding spillover versions of the metrics.'
            ' Adds shifted versions up to the shift value provided.'))
    hyperopt_parser.add_argument(
        '--lme_formula', type=argtypes.StrToTuple(lmeparse, ...), default=(
            lmeparse(
                "RT ~ length + frequency + surprisal + (surprisal||WorkerId)"),
            ),
        help=(
            'Formula for fitting linear mixed effects model for '
            'optimisation for for loglik.'))

    # Fixed Data parser group
    hyperopt_fixed_data_group = hyperopt_parser.add_argument_group('data')
    hyperopt_fixed_data_group.add_argument(
        '--masked', type=argtypes.str_to_bool, default=True,
        help=(
            'Whether to include dependencies in dataset. Necessary for'
            ' dependency parsing setting.'))

    # Flexible Data parser group
    hyperopt_flexible_data_group = hyperopt_parser.add_argument_group('data')
    hyperopt_flexible_data_group.add_argument(
        '--dataset_name', type=argtypes.HyperoptSpace(str),
        help='name of the dataset to load',
        default='Wikitext_processed')
    hyperopt_flexible_data_group.add_argument(
        '--max_len_train',
        type=argtypes.HyperoptSpace(argtypes.OptNone(int)), default=40,
        help='maximum number of tokens in training set')
    hyperopt_flexible_data_group.add_argument(
        '--max_len_eval_test',
        type=argtypes.HyperoptSpace(argtypes.OptNone(int)),
        default=None,
        help='maximum number of tokens in eval set')
    hyperopt_flexible_data_group.add_argument(
        '--min_len_train',
        type=argtypes.HyperoptSpace(argtypes.OptNone(int)), default=3,
        help='minimum number of tokens in training set')
    hyperopt_flexible_data_group.add_argument(
        '--min_len_eval_test',
        type=argtypes.HyperoptSpace(argtypes.OptNone(int)),
        default=None,
        help='minimum number of tokens in eval set')
    hyperopt_flexible_data_group.add_argument(
        '--triangulate', type=argtypes.HyperoptSpace(int), default=0,
        help='TODO')
    hyperopt_flexible_data_group.add_argument(
        '--vocab_size',
        type=argtypes.HyperoptSpace(argtypes.OptNone(int)), default=1_000_000,
        help=(
            'number of most frequent tokens to embed; all other '
            'tokens are replaced with an UNK token;'
            'can be None when loading existing token_mapper'))
    hyperopt_flexible_data_group.add_argument(
        '--first_k',
        type=argtypes.HyperoptSpace(argtypes.OptNone(int)), default=None,
        help='only load first k sentences of the training set')
    hyperopt_flexible_data_group.add_argument(
        '--first_k_eval_test',
        type=argtypes.HyperoptSpace(argtypes.OptNone(int)),
        default=None,
        help='only load first k sentences of the eval and test sets')
    hyperopt_flexible_data_group.add_argument(
        '--connect_with_dummy',
        type=argtypes.HyperoptSpace(argtypes.str_to_bool),
        default=True,
        help=(
            'Establish an arc to a dummy token when there is no '
            'parent/child among the precedents?'))
    hyperopt_flexible_data_group.add_argument(
        '--connect_with_self',
        type=argtypes.HyperoptSpace(argtypes.str_to_bool),
        default=False,
        help=(
            'Establish a recursive arc to the token itself when there '
            'is not parent/child among the precedents?'))
    hyperopt_flexible_data_group.add_argument(
        '--masks_setting', type=argtypes.HyperoptSpace(
            str, choices=("next", "complete", "current")),
        default="current",
        help=('What dependencies to assign to the current token.'))

    # # # Hyperopt Fixed Trainer parser group
    hyperopt_fixed_trainer_group = hyperopt_parser.add_argument_group(
        'trainer_fixed')
    hyperopt_fixed_trainer_group.add_argument(
        '--dependency_mode', type=str,
        choices=("supervised", "input", "standard"),
        default="supervised",
        help="how to use dependency information")
    hyperopt_fixed_trainer_group.add_argument(
        '--combined_loss', type=argtypes.str_to_bool,
        default=False,
        help=(
            "whether to use combined loss"
            " for unsupervised memory cost learning"))
    hyperopt_fixed_trainer_group.add_argument(
        '--batch_size', type=int, default=32,
        help=(
            "batch size; in case of multiple GPUs it is"
            "chunked across the devices"))
    hyperopt_fixed_trainer_group.add_argument(
        '--use_steps', type=argtypes.str_to_bool, default=False,
        help=(
            "Where the unit of evaluation are steps (i.e. batches) "
            "instead of epochs. If true, --eval_interval refers to batches."))
    hyperopt_fixed_trainer_group.add_argument(
        '--max_steps', type=argtypes.OptNone(int), default=100_000,
        help=(
            "maximum number of steps (batches) to process "
            "if --use_steps is set to true"))
    hyperopt_fixed_trainer_group.add_argument(
        '--eval_interval', type=int,
        default=1, help="frequency to perform evaluations in")
    hyperopt_fixed_trainer_group.add_argument(
        '--early_stop_after', type=argtypes.OptNone(int), default=None,
        help=(
            "abort training after x evaluations without"
            "improvement on the eval loss"))
    hyperopt_fixed_trainer_group.add_argument(
        '--early_stop_metric', type=str, default='perplexity',
        help=(
            "metric to chose for early stopping if "
            "--early_stop_after is not none"))
    hyperopt_fixed_trainer_group.add_argument(
        '--epochs', type=int, default=100,
        help="how many epochs to train for")
    hyperopt_fixed_trainer_group.add_argument(
        '--gradient_acc', type=argtypes.OptNone(int), default=None,
        help="If specified, only optimises after n iterations.")

    # # # Hyperopt Flexible Trainer parser group
    hyperopt_flexible_trainer_group = hyperopt_parser.add_argument_group(
        'trainer flexible')
    hyperopt_flexible_trainer_group.add_argument(
        '--learning_rate', type=argtypes.HyperoptSpace(float), default=1e-3,
        help="learning rate for the optimiser")
    hyperopt_flexible_trainer_group.add_argument(
        '--loss_alpha', type=argtypes.HyperoptSpace(float), default=0.5,
        help=(
            "loss weight for supervised learning; 1.0 is only"
            "language model training while 0.0 is only arc training"))
    hyperopt_flexible_trainer_group.add_argument(
        '--distr_mode', type=argtypes.HyperoptSpace(
            argtypes.StrToLiteral("att", "att-n")),
        default="att", choices=("att", "att-n"),
        help="mode for calculation attention distribution for combined loss")
    hyperopt_flexible_trainer_group.add_argument(
        '--global_distr', type=argtypes.HyperoptSpace(
            argtypes.str_to_bool),
        default=True,
        help=(
            "Whether to compute the distribution for the combined loss"
            " globally or as an average of per-head distributions."))
    hyperopt_flexible_trainer_group.add_argument(
        '--include_current', type=argtypes.HyperoptSpace(
            argtypes.str_to_bool),
        default=False,
        help=(
            "Include attention to current item (diagonal) when computing "
            "the attention distribution for attention losses."))
    hyperopt_flexible_trainer_group.add_argument(
        '--length_weighted', type=argtypes.HyperoptSpace(
            argtypes.str_to_bool),
        default=False,
        help=(
            "Normalise attention entropy loss by length of left "
            "context (maximum entropy). The scores are mapped to an "
            "the interval [0, 1]."))
    hyperopt_flexible_trainer_group.add_argument(
        '--losses',
        type=argtypes.OptNone(
            argtypes.HyperoptSpace(argtypes.StrToDict(
                str, argtypes.HyperoptSpace(argtypes.OptNone(float))))),
        default={"lm": 1},
        help=(
            "Dictionary of losses for combined loss setting "
            "and their weights. Must include 'lm'."))
    hyperopt_flexible_trainer_group.add_argument(
        '--k_negatives', type=argtypes.OptNone(
            argtypes.HyperoptSpace(
                argtypes.OptNone(argtypes.HyperoptSpace(int)))),
        default=None,
        help=(
            "k negatives to approximate softmax"))
    hyperopt_flexible_trainer_group.add_argument(
        '--arc_loss_weighted',
        type=argtypes.HyperoptSpace(argtypes.str_to_bool),
        default=False,
        help="Overrepresent arcs against non-arcs in arc loss calculation")
    hyperopt_flexible_trainer_group.add_argument(
        '--discriminative', type=argtypes.HyperoptSpace(argtypes.str_to_bool),
        default=False,
        help="Train the language model in a discriminative fashion")

    # # # Hyperopt Fixed Model parser group
    hyperopt_fixed_model_group = hyperopt_parser.add_argument_group(
        'model fixed')
    hyperopt_fixed_model_group.add_argument(
        '--block_size', type=int, default=500,
        help="maximum sequence length of the model")
    hyperopt_fixed_model_group.add_argument(
        '--overlay_causal', type=argtypes.str_to_bool, default=True,
        help=(
            "whether to overlay a casual mask if providing"
            "masks as additional inputs"))

    # # # Hyperopt Flexible Model parser group
    hyperopt_flexible_model_group = hyperopt_parser.add_argument_group(
        'model flexible')
    hyperopt_flexible_model_group.add_argument(
        '--transformer_description',
        type=argtypes.HyperoptSpace(argtypes.OptNone(TransformerDescription)),
        default=None,  # ((('head', 'child'), 1),),
        help=(
            "Architecture of the transformer model. Tuple of layers "
            "where each layer is a tuple of a tuple of "
            "head types and a width."
            "The width is applied to every head type in the layer."))
    hyperopt_flexible_model_group.add_argument(
        '--layer_design',
        type=argtypes.HyperoptSpace(argtypes.StrToTuple(str, ...)),
        default=None,
        # ("head_current", "child_current"),
        help=(
            "design of the core transformer layer; tuple of head types "
            "is overriden if --transformer_description is provided"))
    hyperopt_flexible_model_group.add_argument(
        '--use_standard',
        type=argtypes.HyperoptSpace(argtypes.str_to_bool), default=False,
        help=("whether to add an unrestricted head to the core layer(s)"))
    hyperopt_flexible_model_group.add_argument(
        '--width',
        type=argtypes.HyperoptSpace(int), default=1,
        help=(
            "width of the core transformer; "
            "is overriden if --transformer_description is provided"))
    hyperopt_flexible_model_group.add_argument(
        '--depth',
        type=argtypes.HyperoptSpace(int), default=1,
        help=(
            "depth of the core transformer; "
            "is overriden if --transformer_description is provided"))
    hyperopt_flexible_model_group.add_argument(
        '--unrestricted_before',
        type=argtypes.HyperoptSpace(int), default=0,
        help=(
            "number of unrestricted layers below the core transformer; "
            "is overriden if --transformer_description is provided"))
    hyperopt_flexible_model_group.add_argument(
        '--unrestricted_after',
        type=argtypes.HyperoptSpace(int), default=0,
        help=(
            "number of unrestricted layers above the core transformer; "
            "is overriden if --transformer_description is provided"))
    hyperopt_flexible_model_group.add_argument(
        '--d_ff_factor', type=argtypes.HyperoptSpace(int), default=4,
        help="hidden dimensionality of the feed-forward layers (*n_embd)")
    hyperopt_flexible_model_group.add_argument(
        '--dropout', type=argtypes.HyperoptSpace(argtypes.OptNone(float)),
        default=0.3,
        help=(
            "hidden dimensionality of the feed-forward layers; "
            "if provided, fixes the search for all specific dropouts;"
            "if provided, specific dropouts establish individual "
            " search spaces"))
    hyperopt_flexible_model_group.add_argument(
        '--dropout_attn', type=argtypes.HyperoptSpace(argtypes.OptNone(float)),
        default=-1,
        help=(
            "dropout for the attention module; "
            "overriden by --dropout if specified"))
    hyperopt_flexible_model_group.add_argument(
        '--dropout_resid',
        type=argtypes.HyperoptSpace(argtypes.OptNone(float)),
        default=-1,
        help=(
            "dropout for the residual connections; "
            "overriden by --dropout if specified"))
    hyperopt_flexible_model_group.add_argument(
        '--dropout_ff', type=argtypes.HyperoptSpace(argtypes.OptNone(float)),
        default=-1,
        help=(
            "dropout for the feed-forward layers; "
            "overriden by --dropout if specified"))
    hyperopt_flexible_model_group.add_argument(
        '--dropout_embd', type=argtypes.HyperoptSpace(argtypes.OptNone(float)),
        default=-1,
        help=(
            "dropout for the embedding layer; "
            "overriden by --dropout if specified"))
    hyperopt_flexible_model_group.add_argument(
        '--dropout_lstm', type=argtypes.HyperoptSpace(argtypes.OptNone(float)),
        default=-1,
        help=(
            "dropout for the LSTM (if existant); "
            "overriden by --dropout if specified"))
    hyperopt_flexible_model_group.add_argument(
        '--use_lstm', type=argtypes.HyperoptSpace(argtypes.str_to_bool),
        default=True,
        help=("use LSTM layer"))
    hyperopt_flexible_model_group.add_argument(
        '--n_embd', type=argtypes.HyperoptSpace(int), default=500,
        help="model embedding size")
    hyperopt_flexible_model_group.add_argument(
        '--use_dual_fixed', type=argtypes.HyperoptSpace(argtypes.str_to_bool),
        default=False,
        help=(
            "whether to cross-fix key and query weights of "
            "the head and child attention heads. Only allowed "
            "if both are present in the description with width 1"))
    hyperopt_flexible_model_group.add_argument(
        '--bias', type=argtypes.HyperoptSpace(argtypes.str_to_bool),
        default=False,
        help="Whether to use bias in all of the model weights")
    hyperopt_flexible_model_group.add_argument(
        '--pos_enc', type=argtypes.HyperoptSpace(
            argtypes.StrToLiteral("embedding", "sinusoidal")),
        default="embedding",
        help="What kind of positional encodings to use.")

    # # Dataprep Parser
    dataprep_parser = subparsers.add_parser(
        "dataprep", help="dataprep mode")

    # Data parser group
    data_group = dataprep_parser.add_argument_group('data')
    data_group.add_argument(
        '--dataset_name', type=str, help='name of the dataset to load',
        default='Wikitext_processed')
    data_group.add_argument(
        '--max_len_train', type=argtypes.OptNone(int), default=40,
        help='maximum number of tokens in training set')
    data_group.add_argument(
        '--max_len_eval_test', type=argtypes.OptNone(int), default=None,
        help='maximum number of tokens in eval set')
    data_group.add_argument(
        '--min_len_train', type=argtypes.OptNone(int), default=3,
        help='minimum number of tokens in training set')
    data_group.add_argument(
        '--min_len_eval_test', type=argtypes.OptNone(int), default=None,
        help='minimum number of tokens in eval set')
    data_group.add_argument(
        '--masked', type=argtypes.str_to_bool, default=True,
        help=(
            'Whether to include dependencies in dataset. Necessary for'
            ' dependency parsing setting.'))
    data_group.add_argument(
        '--triangulate', type=int, default=0,
        help='TODO')
    data_group.add_argument(
        '--vocab_size', type=argtypes.OptNone(int), default=1_000_000,
        help=(
            'number of most frequent tokens to embed; all other '
            'tokens are replaced with an UNK token;'
            'can be None when loading existing token_mapper'))
    data_group.add_argument(
        '--first_k', type=argtypes.OptNone(int), default=None,
        help='only load first k sentences of the training set')
    data_group.add_argument(
        '--first_k_eval_test', type=argtypes.OptNone(int), default=None,
        help='only load first k sentences of the eval and test sets')
    data_group.add_argument(
        '--connect_with_dummy', type=argtypes.str_to_bool, default=True,
        help=(
            'Establish an arc to a dummy token when there is no '
            'parent/child among the precedents?'))
    data_group.add_argument(
        '--connect_with_self', type=argtypes.str_to_bool, default=False,
        help=(
            'Establish a recursive arc to the token itself when there '
            'is not parent/child among the precedents?'))
    data_group.add_argument(
        '--masks_setting', type=str, choices=(
            "complete", "current", "next", "both"),
        default="current",
        help=('What dependencies to assign to the current token.'))

    # # Split Parser
    split_parser = subparsers.add_parser(
        "split", help="split mode")

    # # # Data parser group
    data_group = split_parser.add_argument_group('data')
    data_group.add_argument(
        '--dataset_name', type=str,
        help='name of the psycholinguistic dataset to load',
        choices=data.RTCORPORA,
        default="naturalstories")
    data_group.add_argument(
        '--max_len_train', type=argtypes.OptNone(int), default=Undefined,
        help='Does nothing. TODO')
    data_group.add_argument(
        '--max_len_eval_test', type=argtypes.OptNone(int), default=Undefined,
        help='Does nothing. TODO')
    data_group.add_argument(
        '--min_len_train', type=argtypes.OptNone(int), default=Undefined,
        help='Does nothing. TODO')
    data_group.add_argument(
        '--min_len_eval_test', type=argtypes.OptNone(int), default=Undefined,
        help='Does nothing. TODO')
    data_group.add_argument(
        '--masked', type=argtypes.str_to_bool, default=True,
        help=(
            'Does nothing. TODO'))
    data_group.add_argument(
        '--triangulate', type=int, default=Undefined,
        help='Does nothing. TODO')
    data_group.add_argument(
        '--vocab_size', type=argtypes.OptNone(int), default=Undefined,
        help=(
            'Does nothing. TODO'))
    data_group.add_argument(
        '--first_k', type=argtypes.OptNone(int), default=Undefined,
        help='Does nothing. TODO')
    data_group.add_argument(
        '--first_k_eval_test', type=argtypes.OptNone(int), default=Undefined,
        help='Does nothing. TODO')
    data_group.add_argument(
        '--connect_with_dummy', type=argtypes.str_to_bool, default=Undefined,
        help=(
            'Does nothing. TODO'))
    data_group.add_argument(
        '--connect_with_self', type=argtypes.str_to_bool, default=Undefined,
        help=(
            'Does nothing. TODO'))
    data_group.add_argument(
        '--masks_setting', type=str, choices=(
            "complete", "current", "next", "both"),
        default=Undefined,
        help=('Does nothing. TODO'))

    # # # Functional parser group
    functional_group = split_parser.add_argument_group('functional')
    functional_group.add_argument(
        '--proportion', type=float,
        help=(
            'proportion of corpus to assign to train split'
            '(the remainder is assigned to the test split)'),
        default=0.5)

    # # Test Parser
    test_parser = subparsers.add_parser(
        "test", help="testing mode")

    # # # Data parser group
    data_group = test_parser.add_argument_group('data')
    data_group.add_argument(
        '--dataset_name', type=str, help='name of the dataset to load',
        default=Undefined)
    data_group.add_argument(
        '--max_len_train', type=argtypes.OptNone(int), default=Undefined,
        help='maximum number of tokens in training set')
    data_group.add_argument(
        '--max_len_eval_test', type=argtypes.OptNone(int), default=Undefined,
        help='maximum number of tokens in eval set')
    data_group.add_argument(
        '--min_len_train', type=argtypes.OptNone(int), default=Undefined,
        help='minimum number of tokens in training set')
    data_group.add_argument(
        '--min_len_eval_test', type=argtypes.OptNone(int), default=Undefined,
        help='minimum number of tokens in eval set')
    data_group.add_argument(
        '--masked', type=argtypes.str_to_bool, default=True,
        help=(
            'Whether to include dependencies in dataset. Necessary for '
            'dependency parsing setting.'))
    data_group.add_argument(
        '--triangulate', type=int, default=Undefined,
        help='TODO')
    data_group.add_argument(
        '--vocab_size', type=argtypes.OptNone(int), default=Undefined,
        help=(
            'number of most frequent tokens to embed; all other '
            'tokens are replaced with an UNK token;'
            'can be None when loading existing token_mapper'))
    data_group.add_argument(
        '--first_k', type=argtypes.OptNone(int), default=Undefined,
        help='only load first k sentences of the training set')
    data_group.add_argument(
        '--first_k_eval_test', type=argtypes.OptNone(int), default=Undefined,
        help='only load first k sentences of the eval and test sets')
    data_group.add_argument(
        '--connect_with_dummy', type=argtypes.str_to_bool, default=Undefined,
        help=(
            'Establish an arc to a dummy token when there is no '
            'parent/child among the precedents?'))
    data_group.add_argument(
        '--connect_with_self', type=argtypes.str_to_bool, default=Undefined,
        help=(
            'Establish a recursive arc to the token itself when there '
            'is not parent/child among the precedents?'))
    data_group.add_argument(
        '--masks_setting', type=str, choices=(
            "complete", "current", "next", "both"),
        default=Undefined,
        help=('What dependencies to assign to the current token.'))

    # # # Trainer parser group
    trainer_group = test_parser.add_argument_group('trainer')
    trainer_group.add_argument(
        '--model_name', type=argtypes.OptNone(str),
        default=None,
        help="model name. Set to experiment name if None")
    trainer_group.add_argument(
        '--dependency_mode', type=str,
        choices=("supervised", "input", "standard"),
        default=Undefined,
        help="how to use dependency information")
    trainer_group.add_argument(
        '--combined_loss', type=argtypes.str_to_bool,
        default=Undefined,
        help=(
            "whether to use combined loss for unsupervised"
            " memory cost learning"))
    trainer_group.add_argument(
        '--distr_mode', type=str,
        default=Undefined, choices=("att", "att-n"),
        help="mode for calculation attention distribution for combined loss")
    trainer_group.add_argument(
        '--global_distr', type=argtypes.str_to_bool,
        default=Undefined,
        help=(
            "Whether to compute the distribution for the combined loss"
            " globally or as an average of per-head distributions."))
    trainer_group.add_argument(
        '--include_current', type=argtypes.str_to_bool,
        default=Undefined,
        help=(
            "Include attention to current item (diagonal) when computing"
            "the attention distribution for attention losses."))
    trainer_group.add_argument(
        '--length_weighted', type=argtypes.str_to_bool,
        default=Undefined,
        help=(
            "Normalise attention entropy loss by length of left "
            "context (maximum entropy). The scores are mapped to an "
            "the interval [0, 1]."))
    trainer_group.add_argument(
        '--batch_size', type=int, default=Undefined,
        help=(
            "batch size; in case of multiple GPUs it is "
            "chunked across the devices"))
    trainer_group.add_argument(
        '--loss_alpha', type=argtypes.OptNone(float), default=Undefined,
        help=(
            "loss weight for supervised learning; 1.0 is only "
            "language model training while 0.0 is only arc training"))
    trainer_group.add_argument(
        '--losses', type=argtypes.OptNone(
            argtypes.StrToDict(str, argtypes.OptNone(float))),
        default=Undefined,
        help=(
            "Dictionary of losses for combined loss setting "
            "and their weights. Must include 'lm'."))
    trainer_group.add_argument(
        '--k_negatives', type=argtypes.OptNone(int),
        default=Undefined,
        help=(
            "k negatives to approximate softmax"))
    trainer_group.add_argument(
        '--arc_loss_weighted', type=argtypes.str_to_bool, default=Undefined,
        help="Overrepresent arcs against non-arcs in arc loss calculation")

    # # # Plot parser group
    plot_group = test_parser.add_argument_group('plot')
    plot_group.add_argument(
        '--att_plot', type=argtypes.str_to_bool,
        default=False,
        help="plot attention matrices")
    plot_group.add_argument(
        '--tree_plot', type=argtypes.str_to_bool,
        default=False,
        help="plot dependency trees")

    # # Compare Parser
    compare_parser = subparsers.add_parser(
        "compare", help="comparison mode")
    compare_parser.add_argument(
        '--model1_name', type=str,
        help="name of model 1")
    compare_parser.add_argument(
        '--model2_name', type=str,
        help="name of model 2")
    compare_parser.add_argument(
        '--batch_size', type=int, default=32,
        help=(
            "batch size; in case of multiple GPUs it is "
            "chunked across the devices"))

    # # # Data parser group
    data_group = compare_parser.add_argument_group('data')
    data_group.add_argument(
        '--dataset_name', type=str, help='name of the dataset to load',
        default='Wikitext_processed')

    # # RT Parser
    rt_parser = subparsers.add_parser(
        "rt", help="RT mode")

    settings_group = rt_parser.add_argument_group('settings_group')
    settings_group.add_argument(
        '--lme', type=argtypes.str_to_bool,
        default=True,
        help=("Run linear mixed effects evaluation."))
    settings_group.add_argument(
        '--n_runs', type=int,
        default=1,
        help=(
            "Number of model runs to evaluate."))
    settings_group.add_argument(
        '--legacy_process', type=argtypes.str_to_bool,
        default=False,
        help=(
            "Whether to use the old RT processing function "
            "to replicate results from "
            "https://aclanthology.org/2025.brigap-1.7/"))
    settings_group.add_argument(
        '--to_add', type=argtypes.OptNone(argtypes.StrToTuple(str, ...)),
        default=("surprisal",),
        help=(
            "What predictors to add. Baseline predictors (length, frequency, "
            "position) are always added. Group columns (Corpus, element, word)"
            " are also always available."))

    # # # Data parser group
    data_group = rt_parser.add_argument_group('data')
    data_group.add_argument(
        '--dataset_name', type=str, help='name of the dataset to load',
        choices=data.RTCORPORA,
        default="naturalstories")
    data_group.add_argument(
        '--max_len_train', type=argtypes.OptNone(int), default=None,
        help='Does nothing. TODO')
    data_group.add_argument(
        '--max_len_eval_test', type=argtypes.OptNone(int), default=None,
        help='Does nothing. TODO')
    data_group.add_argument(
        '--min_len_train', type=argtypes.OptNone(int), default=None,
        help='Does nothing. TODO')
    data_group.add_argument(
        '--min_len_eval_test', type=argtypes.OptNone(int), default=None,
        help='Does nothing. TODO')
    data_group.add_argument(
        '--masked', type=argtypes.str_to_bool, default=True,
        help=(
            'Does nothing. TODO'))
    data_group.add_argument(
        '--triangulate', type=int, default=Undefined,
        help='Does nothing. TODO')
    data_group.add_argument(
        '--vocab_size', type=argtypes.OptNone(int), default=Undefined,
        help=(
            'Does nothing. TODO'))
    data_group.add_argument(
        '--first_k', type=argtypes.OptNone(int), default=Undefined,
        help='Does nothing. TODO')
    data_group.add_argument(
        '--first_k_eval_test', type=argtypes.OptNone(int), default=Undefined,
        help='Does nothing. TODO')
    data_group.add_argument(
        '--connect_with_dummy', type=argtypes.str_to_bool, default=Undefined,
        help=(
            'Does nothing. TODO'))
    data_group.add_argument(
        '--connect_with_self', type=argtypes.str_to_bool, default=Undefined,
        help=(
            'Does nothing. TODO'))
    data_group.add_argument(
        '--masks_setting', type=str, choices=(
            "complete", "current", "next", "both"),
        default=Undefined,
        help=('What dependencies to assign to the current token.'))
    data_group.add_argument(
        '--mapper', type=str, default="processed/Wikitext_processed/mapper",
        help='Path to tokeniser. hug:<name> loads a huggingface tokeniser.')

    # # # Trainer parser group
    trainer_group = rt_parser.add_argument_group('trainer')
    trainer_group.add_argument(
        '--model_name', type=argtypes.OptNone(str),
        default=None,
        help=(
            "model name. Is equal to experiment name if None."
            " hug:<name> loads a hugginface model."))
    trainer_group.add_argument(
        '--dependency_mode', type=str,
        choices=("supervised", "input", "standard"),
        default=Undefined,
        help="how to use dependency information")
    trainer_group.add_argument(
        '--combined_loss', type=argtypes.str_to_bool,
        default=Undefined,
        help=(
            "whether to use combined loss for unsupervised"
            " memory cost learning"))
    trainer_group.add_argument(
        '--distr_mode', type=str,
        default=Undefined, choices=("att", "att-n"),
        help="mode for calculation attention distribution for combined loss")
    trainer_group.add_argument(
        '--global_distr', type=argtypes.str_to_bool,
        default=Undefined,
        help=(
            "Whether to compute the distribution for the combined loss"
            " globally or as an average of per-head distributions."))
    trainer_group.add_argument(
        '--include_current', type=argtypes.str_to_bool,
        default=Undefined,
        help=(
            "Include attention to current item (diagonal) when computing"
            "the attention distribution for attention losses."))
    trainer_group.add_argument(
        '--length_weighted', type=argtypes.str_to_bool,
        default=Undefined,
        help=(
            "Normalise attention entropy loss by length of left "
            "context (maximum entropy). The scores are mapped to an "
            "the interval [0, 1]."))
    trainer_group.add_argument(
        '--batch_size', type=int, default=Undefined,
        help=(
            "batch size; in case of multiple GPUs it is "
            "chunked across the devices"))
    trainer_group.add_argument(
        '--loss_alpha', type=argtypes.OptNone(float), default=Undefined,
        help=(
            "loss weight for supervised learning; 1.0 is only "
            "language model training while 0.0 is only arc training"))
    trainer_group.add_argument(
        '--losses', type=argtypes.OptNone(
            argtypes.StrToDict(str, argtypes.OptNone(float))),
        default=Undefined,
        help=(
            "Dictionary of losses for combined loss setting "
            "and their weights. Must include 'lm'."))
    trainer_group.add_argument(
        '--k_negatives', type=argtypes.OptNone(int),
        default=Undefined,
        help=(
            "k negatives to approximate softmax"))
    trainer_group.add_argument(
        '--arc_loss_weighted', type=argtypes.str_to_bool, default=Undefined,
        help="Overrepresent arcs against non-arcs in arc loss calculation")

    # # # Cost parser group
    cost_group = rt_parser.add_argument_group('cost')
    cost_group.add_argument(
        '--shift', type=int, default=0,
        help=(
            'Argument for adding spillover versions of the metrics.'
            ' Adds shifted versions up to the shift value provided.'))
    cost_group.add_argument(
        '--only_content_words_cost', type=argtypes.str_to_bool, default=True,
        help=(
            'Assign costs only to content words?'))
    cost_group.add_argument(
        '--only_content_words_left', type=argtypes.str_to_bool, default=True,
        help=(
            'Take into account only content words in left context when'
            ' computing costs?'))

    return parser
