# python method/VU.py --method simple  --epochs 3 --batch_size 32 --lr 1e-4 --use-reg yes | tee logs/reg_simple.log
# python method/VU.py --method advance  --epochs 3 --batch_size 32 --lr 1e-4 --use-reg yes | tee logs/reg_advance.log
python method/VU.py --method simple  --epochs 3 --batch_size 32 --lr 1e-4 --use-reg no | tee logs/no_reg_simple.log
python method/VU.py --method advance  --epochs 3 --batch_size 32 --lr 1e-4 --use-reg no | tee logs/no_reg_advance.log