python -m method.VU  --method simple  --epochs 1 --batch_size 32 --lr 1e-4 --use-reg no | tee logs/reg_simple.log
python -m method.VU  --method advance  --epochs 1 --batch_size 32 --lr 1e-4 --use-reg no | tee logs/reg_advance.log
