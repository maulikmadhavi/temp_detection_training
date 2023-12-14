# #!/bin/bash

for arch in  yolov7 yolov8;
do
    coverage run  main.py --arch $arch --epochs 3

    mv output output_temp_run2_$arch
    mv .coverage .coverage_run2_$arch
    echo "done $arch"

    echo ""
    echo "--------------------------------------"
    echo ""

done



# # /home/maulik/miniconda3/envs/va-algo/bin/python main.py --arch mobilenetssd --epochs 3
# # mv output output_mobilenetssd

# /home/maulik/miniconda3/envs/va-algo/bin/python main.py --arch yolov7 --epochs 3
# mv output output_yolov7

for arch in  yolov4csp yolov7 yolov8 mobilenetssd yolor;
do
coverage report \
--data-file=.coverage_run2_$arch \
-m `find . | grep .py | grep -v pyc`  > outputreport_run2_$arch.txt
done



coverage combine .coverage_run2_mobilenetssd \
    .coverage_run2_yolor .coverage_run2_yolov4csp \
    .coverage_run2_yolov7 .coverage_run2_yolov8 

coverage report \
--data-file=.coverage \
-m `find . | grep .py | grep -v pyc`  > outputreport.txt    