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
    for batch in train_dataloader:
        outputs = rm_model(**batch, use_cache=False)
        loss = outputs["loss"]
        rm_model.backward(loss)
        rm_model.step()
```
### PPO
```python
for epoch in range(train_epochs):
    for exp_dataset in train_dataloader:
        out += trainer.generate_experience(exp_dataset)
        for ppo_ep in range(ppo_epochs):
            for data in out:
                actor_loss, critic_loss = trainer.train_rlhf(data)
```
```python
def train_rlhf(self, inputs):
            prompts = inputs['prompts']
            log_probs = inputs['logprobs']
            ref_log_probs = inputs['ref_logprobs']
            reward_score = inputs['rewards']
            values = inputs['value']
            attention_mask = inputs['attention_mask']
            seq = inputs['input_ids']
            start = prompts.size()[-1] - 1
            action_mask = attention_mask[:, 1:]

            old_values = values
    with torch.no_grad():
        old_rewards = self.compute_rewards(prompts, log_probs,
                                           ref_log_probs, reward_score,
                                           action_mask)


        advantages, returns = self.get_advantages_and_returns(
            old_values, old_rewards, start)

    ### process the new outputs
    batch = {'input_ids': seq, "attention_mask": attention_mask}
    actor_prob = self.actor_model(**batch, use_cache=False).logits
    actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
    actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                    log_probs[:, start:], advantages,
                                    action_mask[:, start:])
    self.actor_model.backward(actor_loss)
    self.actor_model.step()
    value = self.critic_model.forward_value(**batch,
                                            return_value_only=True,
                                            use_cache=False)[:, :-1]
    critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,
                                                                   start:],
                                      returns, action_mask[:, start:])
    self.critic_model.backward(critic_loss)
    self.critic_model.step()

    return actor_loss, critic_loss
```
```python
def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                    action_mask):

    kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
    rewards = kl_divergence_estimate
    start = prompts.shape[1] - 1
    ends = start + action_mask[:, start:].sum(1) + 1
    reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                              self.clip_reward_value)
    batch_size = log_probs.shape[0]
    for j in range(batch_size):
        rewards[j, start:ends[j]][-1] += reward_clip[j]

    return rewards

def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
    ## policy gradient loss
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                         1.0 + self.cliprange)
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
    return pg_loss

def critic_loss_fn(self, values, old_values, returns, mask):
    ## value loss
    values_clipped = torch.clamp(
        values,
        old_values - self.cliprange_value,
        old_values + self.cliprange_value,
    )
    vf_loss1 = (values - returns)**2
    vf_loss2 = (values_clipped - returns)**2
    vf_loss = 0.5 * torch.sum(
        torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
    return vf_loss

def get_advantages_and_returns(self, values, rewards, start):
    # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
    lastgaelam = 0
    advantages_reversed = []
    length = rewards.size()[-1]
    for t in reversed(range(start, length)):
        nextvalues = values[:, t + 1] if t < length - 1 else 0.0
        delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
        lastgaelam = delta + self.gamma * self.lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values[:, start:]
    return advantages.detach(), returns
```
