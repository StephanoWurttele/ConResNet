python trainPancreasConresnet.py --data_dir=dataset/ --train_list=list/train_list.txt --val_list=list/val_list.txt --snapshot_dir=results/ --input_size=64,120,120 --batch_size=2 --num_gpus=1 --num_steps=40000 --val_pred_every=2000 --learning_rate=1e-4 --num_classes=3 --num_workers=4 --random_mirror=True --random_scale=True