# coverage run train_clean.py --device 1 \
# --batch-size 4 --data /home/maulik/VADE/learning/yolov7/data/data_vitg.yaml \
# --img 320 320 --cfg /home/maulik/VADE/learning/yolov7/cfg/training/yolov7.yaml \
# --weights '' --name yolov7 \
# --hyp /home/maulik/VADE/learning/yolov7/data/hyp.scratch.p5.yaml --epochs 50


# ====== Normal ====
# coverage run train_clean.py --device 1 \
# --batch-size 4 --data /home/maulik/VADE/learning/yolov7/data/data_vitg.yaml \
# --img 320 320 --cfg /home/maulik/VADE/learning/yolov7/cfg/training/yolov7.yaml \
# --weights '' --name yolov7 \
# --hyp /home/maulik/VADE/learning/yolov7/data/hyp.scratch.p5.yaml --epochs 50 \

# === No pastein ===

python train_clean.py --device 1 \
--batch-size 4 --data /home/maulik/VADE/learning/yolov7/data/data_vitg.yaml \
--img 320 320 --cfg /home/maulik/VADE/learning/yolov7/cfg/training/yolov7.yaml \
--weights '' --name yolov7 \
--hyp /home/maulik/VADE/vca-rec-fw/data/hyp.scratch.p5_nopaste_in.yaml --epochs 50
