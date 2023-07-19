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

## process_description
### SFT
```python
for epoch in range(num_train_epochs):
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        model.backward(loss)
        model.step()
```

### RM

## TODO
readme
多机
脚本参数
封装入口脚本
多数据集
prompt
chinese
