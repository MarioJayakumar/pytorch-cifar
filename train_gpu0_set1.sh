
CUDA_VISIBLE_DEVICES=0, python3 main.py --model GoogLeNet  --run 7 --schedule step


CUDA_VISIBLE_DEVICES=0, python3 main.py --model GoogLeNet  --run 8 --schedule exponential


CUDA_VISIBLE_DEVICES=0, python3 main.py --model GoogLeNet  --run 9 --schedule reduce

CUDA_VISIBLE_DEVICES=0, python3 main.py --model GoogLeNet  --run 10 --optim adam
