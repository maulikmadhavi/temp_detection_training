# python main.py --pretrained_model /home/maulik/VADE/learning/yolov7/yolov7_pretrained.pth --arch yolov7

# mv output output_yolov7_pretrained

python main.py --arch yolov7 \
    --pretrained_model /mnt/nas/maulik/tools_backup/VADE/Detection-pretrained/vade_yolov7_pretrain/yolov7_coco_pretrain_dict.pth \
    --epochs 50 \
    --log_path ./output_pretrained_yolov7/log/log.txt \
    --out_dir ./output_pretrained_yolov7/output/ \
    --model_dir ./output_pretrained_yolov7/model/ \
    --checkpoint_dir  ./output_pretrained_yolov7/checkpoint/

