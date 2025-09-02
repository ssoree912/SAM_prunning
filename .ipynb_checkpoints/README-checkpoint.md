# Grouping_pruning_timm

Model preparation: download pre-trained DeiT models for pruning:

```sh
# Download pre-trained DeiT models
sh check_down.sh
```

---

## 1. Exploration


```sh
sh exploration.sh
```

- --attn-prune-rate : Attention pruning rate
- --ffn-prune-rate : FFN pruning rate
- --group_size : pruning_group_size  : (1 or 16)


---

## 2. Fine-Tuning

Unlike exploration phase, use the same script for execution

```sh
sh fine.sh
```
- --checkpoint : Path to model checkpoint from exploration phase

---

## 3. Model compression 

```sh
sh compression.sh
```

- --checkpoint : Path to model checkpoint from exploration and Fine_tuning phases

---
