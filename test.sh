python3 main.py --cfg config/clipclap.yaml \
                --device cuda \
                --root_dir /mnt/c/datasets/ActivityNet \
                --log_dir logs/ClipClap_ActivityNet \
                --dataset_name ActivityNet \
                --epochs 15 \
                --lr 0.0001 \
                --use_wavcaps_embeddings True \
                --modality both \
                --word_embeddings both \
                --run all 

python3 main.py --cfg config/clipclap.yaml \
                --device cuda \
                --root_dir /home/test1/FYP/datasets/UCF \
                --log_dir logs/ClipClap_UCF_MHSA \
                --dataset_name UCF \
                --epochs 20 \
                --lr 0.00007 \
                --use_wavcaps_embeddings True \
                --modality both \
                --word_embeddings both \
                --use_mhsa True \
                --mhsa_num_heads 8 \
                --mhsa_dropout 0.1 \
                --run all

python3 main.py --cfg config/clipclap.yaml \
                        --device cuda \
                        --root_dir /mnt/c/datasets/VGGSound  \
                        --log_dir logs/ClipClap_VGGSound \
                        --dataset_name VGGSound \
                        --epochs 15 \
                        --lr 0.0001 \
                        --use_wavcaps_embeddings True \
                        --modality both  \
                        --word_embeddings both   \
                        --run all > logs/ClipClap_VGGSound.log &
