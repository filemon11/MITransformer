import torch

import optuna
import sys
import argparse
from ast import literal_eval as make_tuple

from mitransformer import io
from mitransformer.utils import logmaker


logger = logmaker.getLogger(__name__)
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


# TODO:correct parser args for data loading
# TODO: make bool args optionally just acccept flag for True


def parse_args() -> (
        io.TrainParserArgs | io.HyperoptParserArgs
        | io.DataprepParserArgs | io.TestParserArgs
        | io.CompareParserArgs):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rank', '--local-rank', type=io.OptNone(int), default=None,
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
        '--use_ddp', type=io.str_to_bool,
        default=torch.cuda.device_count() > 1,
        help="whether to use distributed GPU training")
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
        '--max_len_train', type=io.OptNone(int), default=40,
        help='maximum number of tokens in training set')
    data_group.add_argument(
        '--max_len_eval_test', type=io.OptNone(int), default=None,
        help='maximum number of tokens in eval set')
    data_group.add_argument(
        '--triangulate', type=int, default=0,
        help='TODO')
    data_group.add_argument(
        '--vocab_size', type=io.OptNone(int), default=50_000,
        help=(
            'number of most frequent tokens to embed; all other '
            'tokens are replaced with an UNK token;'
            'can be None when loading existing token_mapper'))
    data_group.add_argument(
        '--first_k', type=io.OptNone(int), default=None,
        help='only load first k sentences of the training set')
    data_group.add_argument(
        '--first_k_eval_test', type=io.OptNone(int), default=None,
        help='only load first k sentences of the eval and test sets')
    data_group.add_argument(
        '--connect_with_dummy', type=io.str_to_bool, default=True,
        help=(
            'Establish an arc to a dummy token when there is no '
            'parent/child among the precedents?'))
    data_group.add_argument(
        '--connect_with_self', type=io.str_to_bool, default=False,
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
        '--model_name', type=io.OptNone(str),
        default=None,
        help="model name. Set to experiment name if None")
    trainer_group.add_argument(
        '--dependency_mode', type=str,
        choices=("supervised", "input", "standard"),
        default="supervised",
        help="how to use dependency information")
    trainer_group.add_argument(
        '--batch_size', type=int, default=32,
        help=(
            "batch size; in case of multiple GPUs it is "
            "chunked across the devices"))
    trainer_group.add_argument(
        '--use_steps', type=io.str_to_bool, default=False,
        help=(
            "Where the unit of evaluation are steps (i.e. batches) "
            "instead of epochs. If true, --eval_interval "
            "refers to number of batches processed."))
    trainer_group.add_argument(
        '--max_steps', type=io.OptNone(int), default=100_000,
        help=(
            "maximum number of steps (batches) to process "
            "if --use_steps is set to true"))
    trainer_group.add_argument(
        '--eval_interval', type=int,
        default=1, help="frequency to perform evaluations in")
    trainer_group.add_argument(
        '--early_stop_after', type=io.OptNone(int), default=None,
        help=(
            "abort training after x evaluations without "
            "improvement on the eval loss; if None, no early stopping"))
    trainer_group.add_argument(
        '--early_stop_metric', type=io.OptNone(str), default='perplexity',
        help=(
            "metric to chose for early stopping if "
            "--early_stop_after is not none"))
    trainer_group.add_argument(
        '--epochs', type=int, default=100,
        help="how many epochs to train for")
    trainer_group.add_argument(
        '--gradient_acc', type=io.OptNone(int), default=None,
        help="If specified, only optimises after n iterations.")
    trainer_group.add_argument(
        '--learning_rate', type=float, default=1e-3,
        help="learning rate for the optimiser")
    trainer_group.add_argument(
        '--loss_alpha', type=io.OptNone(float), default=0.5,
        help=(
            "loss weight for supervised learning; 1.0 is only "
            "language model training while 0.0 is only arc training"))
    trainer_group.add_argument(
        '--arc_loss_weighted', type=io.str_to_bool, default=False,
        help="Overrepresent arcs against non-arcs in arc loss calculation")
    trainer_group.add_argument(
        '--discriminative', type=io.str_to_bool, default=False,
        help="Train the language model in a discriminative fashion")

    # # # Model parser group
    model_group = train_parser.add_argument_group('model')
    model_group.add_argument(
        '--transformer_description',
        type=io.OptNone(make_tuple), default=None,
        help=(
            "Architecture of the transformer model. Tuple of layers "
            "where each layer is a tuple of a tuple of "
            "head types and a width."
            "The width is applied to every head type in the layer."
            "If provised, overrides --layer_design, --use_standard, "
            "--width, --depth, --unrestricted_before, --unrestricted_after"))
    model_group.add_argument(
        '--layer_design',
        type=io.OptNone(make_tuple), default=None,
        # ("head_current", "child_current"),
        help=(
            "design of the core transformer layer; tuple of head types "
            "is overriden if --transformer_description is provided"))
    model_group.add_argument(
        '--use_standard',
        type=io.str_to_bool, default=False,
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
        '--dropout', type=io.OptNone(float), default=0.3,
        help=(
            "hidden dimensionality of the feed-forward layers; "
            "can be further specified by additional dropout params"))
    model_group.add_argument(
        '--dropout_attn', type=io.OptNone(float), default=0,
        help=(
            "dropout for the attention module; "
            "overridden by --dropout if set to -1"))
    model_group.add_argument(
        '--dropout_resid', type=io.OptNone(float), default=-1,
        help=(
            "dropout for the residual connections; "
            "overridden by --dropout if set to -1"))
    model_group.add_argument(
        '--dropout_ff', type=io.OptNone(float), default=-1,
        help=(
            "dropout for the feed-forward layers; "
            "overridden by --dropout if set to -1"))
    model_group.add_argument(
        '--dropout_embd', type=io.OptNone(float), default=-1,
        help=(
            "dropout for the embedding layer; "
            "overridden by --dropout if set to -1"))
    model_group.add_argument(
        '--dropout_lstm', type=io.OptNone(float), default=-1,
        help=(
            "dropout for the LSTM (if existant); "
            "overridden by --dropout if set to -1"))
    model_group.add_argument(
        '--use_lstm', type=io.str_to_bool, default=True,
        help=("use LSTM layer"))
    model_group.add_argument(
        '--block_size', type=int, default=500,
        help="maximum sequence length of the model")
    model_group.add_argument(
        '--n_embd', type=int, default=500,
        help="model embedding size")
    model_group.add_argument(
        '--overlay_causal', type=io.str_to_bool, default=True,
        help=(
            "whether to overlay a casual mask if providing"
            "masks as additional inputs"))
    model_group.add_argument(
        '--use_dual_fixed', type=io.str_to_bool, default=False,
        help=(
            "whether to cross-fix key and query weights of "
            "the head and child attention heads. Only allowed "
            "if both are present in the description with width 1"))
    model_group.add_argument(
        '--bias', type=io.str_to_bool, default=False,
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
        choices=("perplexity", "uas", "loss", "lm_loss", "arc_loss"),
        default="perplexity",
        help="metric to optimise")
    hyperopt_parser.add_argument(
        '--n_warmup_steps', type=int,
        default=1,
        help=(
            "how many evaluations to wait before pruning can "
            "happen within a trial"))
    hyperopt_parser.add_argument(
        '--n_startup_trials', type=int,
        default=5,
        help=(
            "how many trials to run before pruning can happen at all. "))
    hyperopt_parser.add_argument(
        '--n_trials', type=int,
        default=25,
        help="how many trials to run")

    # Data parser group
    data_group = hyperopt_parser.add_argument_group('data')
    data_group.add_argument(
        '--dataset_name', type=io.io.HyperoptSpace(str),
        help='name of the dataset to load',
        default='Wikitext_processed')
    data_group.add_argument(
        '--max_len_train', type=io.HyperoptSpace(io.OptNone(int)), default=40,
        help='maximum number of tokens in training set')
    data_group.add_argument(
        '--max_len_eval_test', type=io.HyperoptSpace(io.OptNone(int)),
        default=None,
        help='maximum number of tokens in eval set')
    data_group.add_argument(
        '--triangulate', type=io.HyperoptSpace(int), default=0,
        help='TODO')
    data_group.add_argument(
        '--vocab_size', type=io.HyperoptSpace(io.OptNone(int)), default=50_000,
        help=(
            'number of most frequent tokens to embed; all other '
            'tokens are replaced with an UNK token;'
            'can be None when loading existing token_mapper'))
    data_group.add_argument(
        '--first_k', type=io.HyperoptSpace(io.OptNone(int)), default=None,
        help='only load first k sentences of the training set')
    data_group.add_argument(
        '--first_k_eval_test', type=io.HyperoptSpace(io.OptNone(int)),
        default=None,
        help='only load first k sentences of the eval and test sets')
    data_group.add_argument(
        '--connect_with_dummy', type=io.HyperoptSpace(io.str_to_bool),
        default=True,
        help=(
            'Establish an arc to a dummy token when there is no '
            'parent/child among the precedents?'))
    data_group.add_argument(
        '--connect_with_self', type=io.HyperoptSpace(io.str_to_bool),
        default=False,
        help=(
            'Establish a recursive arc to the token itself when there '
            'is not parent/child among the precedents?'))
    data_group.add_argument(
        '--masks_setting', type=io.HyperoptSpace(
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
        '--batch_size', type=int, default=32,
        help=(
            "batch size; in case of multiple GPUs it is"
            "chunked across the devices"))
    hyperopt_fixed_trainer_group.add_argument(
        '--use_steps', type=io.str_to_bool, default=False,
        help=(
            "Where the unit of evaluation are steps (i.e. batches) "
            "instead of epochs. If true, --eval_interval refers to batches."))
    hyperopt_fixed_trainer_group.add_argument(
        '--max_steps', type=io.OptNone(int), default=100_000,
        help=(
            "maximum number of steps (batches) to process "
            "if --use_steps is set to true"))
    hyperopt_fixed_trainer_group.add_argument(
        '--eval_interval', type=int,
        default=1, help="frequency to perform evaluations in")
    hyperopt_fixed_trainer_group.add_argument(
        '--early_stop_after', type=io.OptNone(int), default=None,
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
        '--gradient_acc', type=io.OptNone(int), default=None,
        help="If specified, only optimises after n iterations.")

    # # # Hyperopt Flexible Trainer parser group
    hyperopt_flexible_trainer_group = hyperopt_parser.add_argument_group(
        'trainer flexible')
    hyperopt_flexible_trainer_group.add_argument(
        '--learning_rate', type=io.HyperoptSpace(float), default=1e-3,
        help="learning rate for the optimiser")
    hyperopt_flexible_trainer_group.add_argument(
        '--loss_alpha', type=io.HyperoptSpace(float), default=0.5,
        help=(
            "loss weight for supervised learning; 1.0 is only"
            "language model training while 0.0 is only arc training"))
    hyperopt_flexible_trainer_group.add_argument(
        '--arc_loss_weighted', type=io.HyperoptSpace(io.str_to_bool),
        default=False,
        help="Overrepresent arcs against non-arcs in arc loss calculation")
    hyperopt_flexible_trainer_group.add_argument(
        '--discriminative', type=io.HyperoptSpace(io.str_to_bool),
        default=False,
        help="Train the language model in a discriminative fashion")

    # # # Hyperopt Fixed Model parser group
    hyperopt_fixed_model_group = hyperopt_parser.add_argument_group(
        'model fixed')
    hyperopt_fixed_model_group.add_argument(
        '--block_size', type=int, default=500,
        help="maximum sequence length of the model")
    hyperopt_fixed_model_group.add_argument(
        '--overlay_causal', type=io.str_to_bool, default=True,
        help=(
            "whether to overlay a casual mask if providing"
            "masks as additional inputs"))

    # # # Hyperopt Flexible Model parser group
    hyperopt_flexible_model_group = hyperopt_parser.add_argument_group(
        'model flexible')
    hyperopt_flexible_model_group.add_argument(
        '--transformer_description',
        type=io.HyperoptSpace(io.OptNone(make_tuple)),
        default=None,  # ((('head', 'child'), 1),),
        help=(
            "Architecture of the transformer model. Tuple of layers "
            "where each layer is a tuple of a tuple of "
            "head types and a width."
            "The width is applied to every head type in the layer."))
    hyperopt_flexible_model_group.add_argument(
        '--layer_design',
        type=io.HyperoptSpace(make_tuple), default=None,
        # ("head_current", "child_current"),
        help=(
            "design of the core transformer layer; tuple of head types "
            "is overriden if --transformer_description is provided"))
    hyperopt_flexible_model_group.add_argument(
        '--use_standard',
        type=io.HyperoptSpace(io.str_to_bool), default=False,
        help=("whether to add an unrestricted head to the core layer(s)"))
    hyperopt_flexible_model_group.add_argument(
        '--width',
        type=io.HyperoptSpace(int), default=1,
        help=(
            "width of the core transformer; "
            "is overriden if --transformer_description is provided"))
    hyperopt_flexible_model_group.add_argument(
        '--depth',
        type=io.HyperoptSpace(int), default=1,
        help=(
            "depth of the core transformer; "
            "is overriden if --transformer_description is provided"))
    hyperopt_flexible_model_group.add_argument(
        '--unrestricted_before',
        type=io.HyperoptSpace(int), default=0,
        help=(
            "number of unrestricted layers below the core transformer; "
            "is overriden if --transformer_description is provided"))
    hyperopt_flexible_model_group.add_argument(
        '--unrestricted_after',
        type=io.HyperoptSpace(int), default=0,
        help=(
            "number of unrestricted layers above the core transformer; "
            "is overriden if --transformer_description is provided"))
    hyperopt_flexible_model_group.add_argument(
        '--d_ff_factor', type=io.HyperoptSpace(int), default=4,
        help="hidden dimensionality of the feed-forward layers (*n_embd)")
    hyperopt_flexible_model_group.add_argument(
        '--dropout', type=io.HyperoptSpace(io.OptNone(float)), default=0.3,
        help=(
            "hidden dimensionality of the feed-forward layers; "
            "if provided, fixes the search for all specific dropouts;"
            "if provided, specific dropouts establish individual "
            " search spaces"))
    hyperopt_flexible_model_group.add_argument(
        '--dropout_attn', type=io.HyperoptSpace(io.OptNone(float)), default=-1,
        help=(
            "dropout for the attention module; "
            "overriden by --dropout if specified"))
    hyperopt_flexible_model_group.add_argument(
        '--dropout_resid', type=io.HyperoptSpace(io.OptNone(float)),
        default=-1,
        help=(
            "dropout for the residual connections; "
            "overriden by --dropout if specified"))
    hyperopt_flexible_model_group.add_argument(
        '--dropout_ff', type=io.HyperoptSpace(io.OptNone(float)), default=-1,
        help=(
            "dropout for the feed-forward layers; "
            "overriden by --dropout if specified"))
    hyperopt_flexible_model_group.add_argument(
        '--dropout_embd', type=io.HyperoptSpace(io.OptNone(float)), default=-1,
        help=(
            "dropout for the embedding layer; "
            "overriden by --dropout if specified"))
    hyperopt_flexible_model_group.add_argument(
        '--dropout_lstm', type=io.HyperoptSpace(io.OptNone(float)), default=-1,
        help=(
            "dropout for the LSTM (if existant); "
            "overriden by --dropout if specified"))
    hyperopt_flexible_model_group.add_argument(
        '--use_lstm', type=io.HyperoptSpace(io.str_to_bool), default=True,
        help=("use LSTM layer"))
    hyperopt_flexible_model_group.add_argument(
        '--n_embd', type=io.HyperoptSpace(int), default=500,
        help="model embedding size")
    hyperopt_flexible_model_group.add_argument(
        '--use_dual_fixed', type=io.HyperoptSpace(io.str_to_bool),
        default=False,
        help=(
            "whether to cross-fix key and query weights of "
            "the head and child attention heads. Only allowed "
            "if both are present in the description with width 1"))
    hyperopt_flexible_model_group.add_argument(
        '--bias', type=io.HyperoptSpace(io.str_to_bool), default=False,
        help="Whether to use bias in all of the model weights")
    hyperopt_flexible_model_group.add_argument(
        '--pos_enc', type=io.HyperoptSpace(
            io.StrToLiteral("embedding", "sinusoidal")),
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
        '--max_len_train', type=io.OptNone(int), default=40,
        help='maximum number of tokens in training set')
    data_group.add_argument(
        '--max_len_eval_test', type=io.OptNone(int), default=None,
        help='maximum number of tokens in eval set')
    data_group.add_argument(
        '--triangulate', type=int, default=0,
        help='TODO')
    data_group.add_argument(
        '--vocab_size', type=io.OptNone(int), default=50_000,
        help=(
            'number of most frequent tokens to embed; all other '
            'tokens are replaced with an UNK token;'
            'can be None when loading existing token_mapper'))
    data_group.add_argument(
        '--first_k', type=io.OptNone(int), default=None,
        help='only load first k sentences of the training set')
    data_group.add_argument(
        '--first_k_eval_test', type=io.OptNone(int), default=None,
        help='only load first k sentences of the eval and test sets')
    data_group.add_argument(
        '--connect_with_dummy', type=io.str_to_bool, default=True,
        help=(
            'Establish an arc to a dummy token when there is no '
            'parent/child among the precedents?'))
    data_group.add_argument(
        '--connect_with_self', type=io.str_to_bool, default=False,
        help=(
            'Establish a recursive arc to the token itself when there '
            'is not parent/child among the precedents?'))
    data_group.add_argument(
        '--masks_setting', type=str, choices=(
            "complete", "current", "next", "both"),
        default="current",
        help=('What dependencies to assign to the current token.'))

    # # Test Parser
    test_parser = subparsers.add_parser(
        "test", help="testing mode")

    # # # Data parser group
    data_group = test_parser.add_argument_group('data')
    data_group.add_argument(
        '--dataset_name', type=str, help='name of the dataset to load',
        default=io.Undefined)
    data_group.add_argument(
        '--max_len_train', type=io.OptNone(int), default=io.Undefined,
        help='maximum number of tokens in training set')
    data_group.add_argument(
        '--max_len_eval_test', type=io.OptNone(int), default=io.Undefined,
        help='maximum number of tokens in eval set')
    data_group.add_argument(
        '--triangulate', type=int, default=io.Undefined,
        help='TODO')
    data_group.add_argument(
        '--vocab_size', type=io.OptNone(int), default=io.Undefined,
        help=(
            'number of most frequent tokens to embed; all other '
            'tokens are replaced with an UNK token;'
            'can be None when loading existing token_mapper'))
    data_group.add_argument(
        '--first_k', type=io.OptNone(int), default=io.Undefined,
        help='only load first k sentences of the training set')
    data_group.add_argument(
        '--first_k_eval_test', type=io.OptNone(int), default=io.Undefined,
        help='only load first k sentences of the eval and test sets')
    data_group.add_argument(
        '--connect_with_dummy', type=io.str_to_bool, default=io.Undefined,
        help=(
            'Establish an arc to a dummy token when there is no '
            'parent/child among the precedents?'))
    data_group.add_argument(
        '--connect_with_self', type=io.str_to_bool, default=io.Undefined,
        help=(
            'Establish a recursive arc to the token itself when there '
            'is not parent/child among the precedents?'))
    data_group.add_argument(
        '--masks_setting', type=str, choices=(
            "complete", "current", "next", "both"),
        default=io.Undefined,
        help=('What dependencies to assign to the current token.'))

    # # # Trainer parser group
    trainer_group = test_parser.add_argument_group('trainer')
    trainer_group.add_argument(
        '--model_name', type=io.OptNone(str),
        default=None,
        help="model name. Set to experiment name if None")
    trainer_group.add_argument(
        '--dependency_mode', type=str,
        choices=("supervised", "input", "standard"),
        default=io.Undefined,
        help="how to use dependency information")
    trainer_group.add_argument(
        '--batch_size', type=int, default=io.Undefined,
        help=(
            "batch size; in case of multiple GPUs it is "
            "chunked across the devices"))
    trainer_group.add_argument(
        '--loss_alpha', type=io.OptNone(float), default=io.Undefined,
        help=(
            "loss weight for supervised learning; 1.0 is only "
            "language model training while 0.0 is only arc training"))
    trainer_group.add_argument(
        '--arc_loss_weighted', type=io.str_to_bool, default=io.Undefined,
        help="Overrepresent arcs against non-arcs in arc loss calculation")

    # # # Plot parser group
    plot_group = test_parser.add_argument_group('plot')
    plot_group.add_argument(
        '--att_plot', type=io.str_to_bool,
        default=False,
        help="plot attention matrices")
    plot_group.add_argument(
        '--tree_plot', type=io.str_to_bool,
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

    args = parser.parse_args()
    match args.mode:
        case "train":
            return io.TrainParserArgs(**vars(args))
        case "hyperopt":
            return io.HyperoptParserArgs(**vars(args))
        case "dataprep":
            return io.DataprepParserArgs(**vars(args))
        case "test":
            return io.TestParserArgs(**vars(args))
        case _:
            return io.CompareParserArgs(**vars(args))


if __name__ == "__main__":
    args: (
        io.TrainParserArgs | io.HyperoptParserArgs
        | io.DataprepParserArgs | io.TestParserArgs
        | io.CompareParserArgs)
    args = parse_args()
    logmaker.logging_config(logname=args.name)
    # logging_config(
    #     logname="log",
    #     logpath=os.path.join(LMTrainer.model_dir, name))
    #  TODO: save trainer and model log in model dir
    #  but general and hyperopt log in normal logdir

    # tokenise dataset
    # TODO: do this automatically
    # problem: cannot do in parallel and
    # therefore leads to timeout when doing
    # on only one rank.

    io.args_logic(args)

    # For hyperopt we deal with dropout later

    # Some checks
    assert not (args.use_ddp and args.device == "cpu"), (
        "Must set --device to a GPU when setting --use_ddp. "
        f"Received --device {args.device}, --use_ddp {args.use_ddp}")

    logmaker.info(args.rank, logger, f"Arguments provided: {str(sys.argv)}")
    io.main(args)
    # if n_devices > 1:
    #     mp.spawn(main, args=(n_devices,), nprocs=n_devices)  # type: ignore
    # else:
    #     main(None, n_devices)
