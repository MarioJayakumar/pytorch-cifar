
CUDA_VISIBLE_DEVICES=1, python3 main.py --model resnet101  --run 7 --schedule step

CUDA_VISIBLE_DEVICES=1, python3 main.py --model resnet101  --run 8 --schedule exponential

CUDA_VISIBLE_DEVICES=1, python3 main.py --model resnet101  --run 9 --schedule reduce

CUDA_VISIBLE_DEVICES=1, python3 main.py --model resnet101  --run 10 --optim adam