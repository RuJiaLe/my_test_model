import argparse

parser = argparse.ArgumentParser()

# parameters
parser.add_argument('--total_epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--size', type=int, default=256, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate of weight')
parser.add_argument('--decay_epoch', type=int, default=15, help='every n epochs decay learning rate')
parser.add_argument('--log_dir', type=str, default="./Log_file", help="log_dir file")
parser.add_argument('--start_epoch', type=int, default=1, help='start_epoch')
parser.add_argument("--clip_len", type=int, default=5, help="the number of frames in a video clip.")

# data_path
parser.add_argument('--image_train_path', type=str, default="./data/Image_train_data", help='image_train_path')
parser.add_argument('--image_val_path', type=str, default="./data/Image_val_data", help='image_val_path')

parser.add_argument('--video_train_path', type=str, default="./data/Video_train_data", help='video_train_path')
parser.add_argument('--video_val_path', type=str, default="./data/Video_val_data", help='video_val_path')

parser.add_argument('--predict_data_path', type=str, default="./data/Video_test_data", help='predict_data_path')

# dataset
parser.add_argument('--image_train_dataset', type=list, default=["DUTS"], help='image_train_dataset')
parser.add_argument('--image_val_dataset', type=list, default=["DUTS"], help='image_val_dataset')

parser.add_argument('--video_train_dataset', type=list, default=["DAVSOD_111", "DAVIS_30", "UVSD_9"], help='video_train_dataset')
parser.add_argument('--video_val_dataset', type=list, default=["Val_data"], help='video_val_dataset')

parser.add_argument('--predict_dataset', type=str, default="DAVIS_20",
                    choices=["DAVIS_20", "DAVSOD_Difficult_15", "DAVSOD_Easy_25", "DAVSOD_Normal_15", "DAVSOD_Validation_Set_21", "UVSD_9"],
                    help='predict_dataset')

# model_path
parser.add_argument('--model_path', type=str, default="./save_models", help='save_model_path')

args = parser.parse_args()
