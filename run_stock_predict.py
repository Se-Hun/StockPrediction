import os
import argparse

from utils import is_gpu_available, HParams

def main():
    parser = argparse.ArgumentParser()

    # Required Parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, help="bert OR lstm", required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output_result directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_mode", default="regression", type=str,
                        help="classification or regression", required=True)

    # Other Parameters
    parser.add_argument("--use_gpu", help="use gpu=True or False", default=True)
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial classifier rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of classifier epochs to perform.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for classifier.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for classifier.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run classifier.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")

    args = parser.parse_args()

    # use GPU ?
    if not is_gpu_available():
        args.use_gpu = False

    # Model
    model_type = args.model_type
    output_mode = args.output_mode

    # data
    data_root = args.data_dir

    # output
    output_root = args.output_dir

    # Train
    if args.do_train:
        hps = HParams(
            # gpu setting ----------------------------------------
            use_gpu = args.use_gpu,

            # train settings ----------------------------------------
            learning_rate = args.learning_rate,
            num_train_epochs = args.num_train_epochs,
            per_gpu_train_batch_size = args.per_gpu_train_batch_size,

            # model settings ----------------------------------------
            model_type = model_type,
            output_mode = output_mode
        )

        hps.show()

        print("*********** Start Training ***********")
        run_train()

    if args.do_eval:
        print("*********** Start Evaluating ***********")
        run_eval()


if __name__ == "__main__":
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    main()

