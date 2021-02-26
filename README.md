# SUMO-MARL

```
Implementation of Coordinated Q-learning traffic lights. 


Adapted from Lucas Alegre's github repo: https://github.com/LucasAlegre/sumo-rl
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

## Example Networks
```
Single intersection.
A grid of 4 traffic lights arranged 2 by 2.
A grid of 9 traffic lights arranged 3 by 3.
A grid of 64 traffic lights arranged 8 by 8.
```

## Implemented Algorithms
```
Q-learning.
Coordinated Q-learning (with variable elimination).
Deep Q-learning.
```


### [Q-learning] in a single intersection:
```
python3 experiments/q_singleintersection.py -tripfile outputs/qtrip.xml
```

### To plot results:
```
python3 outputs/plot.py -f outputs/my-single-intersection/q
```

