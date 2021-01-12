
CUDA_VISIBLE_DEVICES=1, python3 main.py --model resnet101  --run 11 --schedule step

CUDA_VISIBLE_DEVICES=1, python3 main.py --model resnet101  --run 12 --schedule exponential

CUDA_VISIBLE_DEVICES=1, python3 main.py --model resnet101  --run 13 --schedule reduce