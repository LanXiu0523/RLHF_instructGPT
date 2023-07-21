# instructGPT
Reproduce instructGPT

## Install
```
git clone 
cd

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
source venv/bin/activate
```
X单机单卡：
```bash
bash train.sh sgl_gpu
```
V单机多卡：
```bash
bash train.sh sgl_mach
```

## process_description
### SFT
```python
for epoch in range(train_epochs):
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        model.backward(loss)
        model.step()
```
### RM
```python
for epoch in range(train_epochs):
    rm_model.train()
    mean_loss = 0
    for batch in train_dataloader:
        outputs = rm_model(**batch, use_cache=False)
        loss = outputs["loss"]
        rm_model.backward(loss)
        rm_model.step()
        mean_loss += loss.item()
```
### PPO
```python
for epoch in range(train_epochs):
    for exp_dataset in train_dataloader:
        inner_iter = 0
        actor_loss_sum, critic_loss_sum, unsup_loss_sum = 0, 0, 0
        average_reward = 0
        for ppo_ep in range(ppo_epochs):
            for exp_data in enumerate exp_dataset:
                actor_loss, critic_loss = trainer.train_rlhf(exp_data)
                actor_loss_sum += actor_loss.item()
                critic_loss_sum += critic_loss.item()
                average_reward += exp_data["rewards"].mean()
                inner_iter += 1
```
