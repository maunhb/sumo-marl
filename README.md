# SUMO-MARL

```
Based off Lucas Alegre's github: https://github.com/LucasAlegre/sumo-rl
```

## Install

### To install SUMO v1.2.0:

```
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc 
```
Don't forget to set SUMO_HOME variable (default sumo installation path is /usr/share/sumo)
```
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```

### To install sumo_rl package:
```
pip3 install -e .
```

## Examples

Check [experiments] to see how to instantiate a SumoEnvironment and use it with your RL algorithm.

### [Q-learning] in a single intersection:
```
python3 experiments/q_singleintersection.py -tripfile outputs/qtrip.xml
```

### To plot results:
```
python3 outputs/plot.py -f outputs/my-single-intersection/q
```

