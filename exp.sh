# some tasks need a pre-generated goal.
python fluidlab/run.py --cfg_file configs/exp_latteart.yaml --record --renderer_type GGUI
python fluidlab/run.py --cfg_file configs/exp_latteart.yaml --record
python fluidlab/run.py --cfg_file configs/exp_latteart_stir.yaml --record
python fluidlab/run.py --cfg_file configs/exp_icecream.yaml --record
python fluidlab/run.py --cfg_file configs/exp_icecream_simple.yaml --record

# solve
python fluidlab/run.py --cfg_file configs/exp_latteart.yaml --exp_name exp_latteart
python fluidlab/run.py --cfg_file configs/exp_latteart_stir.yaml --exp_name exp_latteart_stir
python fluidlab/run.py --cfg_file configs/exp_icecream.yaml --exp_name exp_icecream
python fluidlab/run.py --cfg_file configs/exp_gathering.yaml --exp_name exp_gathering
python fluidlab/run.py --cfg_file configs/exp_gatheringO.yaml --exp_name exp_gatheringO
python fluidlab/run.py --cfg_file configs/exp_circulation.yaml --exp_name exp_circulation
python fluidlab/run.py --cfg_file configs/exp_pouring.yaml --exp_name exp_pouring
python fluidlab/run.py --cfg_file configs/exp_icecream_simple.yaml --exp_name exp_icecream_simple
python fluidlab/run.py --cfg_file configs/exp_mixing.yaml --exp_name exp_mixing
python fluidlab/run.py --cfg_file configs/exp_transporting.yaml --exp_name exp_transporting



