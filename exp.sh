# some tasks need a pre-generated goal.
python fluidlab/run.py --cfg_file configs/exp_latteart.yaml --record
python fluidlab/run.py --cfg_file configs/exp_icecream_static.yaml --record
python fluidlab/run.py --cfg_file configs/exp_icecream_dynamic.yaml --record
python fluidlab/run.py --cfg_file configs/exp_latteart_stir.yaml --record --user_input

# solve
python fluidlab/run.py --cfg_file configs/exp_latteart.yaml --exp_name exp_latteart
python fluidlab/run.py --cfg_file configs/exp_latteart_stir.yaml --exp_name exp_latteart_stir
python fluidlab/run.py --cfg_file configs/exp_icecream_static.yaml --exp_name exp_icecream_static
python fluidlab/run.py --cfg_file configs/exp_icecream_dynamic.yaml --exp_name exp_icecream_dynamic
python fluidlab/run.py --cfg_file configs/exp_gathering_easy.yaml --exp_name exp_gathering_easy 

python fluidlab/run.py --cfg_file configs/exp_gatheringO.yaml --exp_name exp_gatheringO
python fluidlab/run.py --cfg_file configs/exp_circulation.yaml --exp_name exp_circulation
python fluidlab/run.py --cfg_file configs/exp_pouring.yaml --exp_name exp_pouring
python fluidlab/run.py --cfg_file configs/exp_mixing.yaml --exp_name exp_mixing
python fluidlab/run.py --cfg_file configs/exp_transporting.yaml --exp_name exp_transporting

python fluidlab/run.py --cfg_file configs/exp_gathering_easy.yaml --replay_policy --path logs/policies/exp_gathering_easy/0499.pkl



