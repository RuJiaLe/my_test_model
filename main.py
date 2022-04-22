from torch.utils.data import DataLoader
from utils.dataload import VideoDataset
from utils.transforms import get_train_transforms, get_transforms

from utils.model import Model
from utils.Train_material import start_train
from utils.config import args


def train():
    print("start video training!!!")
    model = Model()

    train_transforms = get_train_transforms(input_size=(args.size, args.size))
    train_dataset = VideoDataset(root_dir=args.video_train_path, train_set_list=args.video_train_dataset, training=True,
                                 transforms=train_transforms, clip_len=args.clip_len)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    # val data load
    val_transforms = get_transforms(input_size=(args.size, args.size))
    val_dataset = VideoDataset(root_dir=args.video_val_path, train_set_list=args.video_val_dataset, training=True,
                               transforms=val_transforms, clip_len=args.clip_len)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    print(f'load train data done, total train number is {len(train_dataloader) * args.clip_len * args.batch_size}')
    print(f'load val data done, total val number is {len(val_dataloader) * args.clip_len * args.batch_size}')

    start_train(train_dataloader, val_dataloader, model)

    print('-------------Congratulations! Training Done!!!-------------')


if __name__ == '__main__':
    train()




    
