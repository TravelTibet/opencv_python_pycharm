- 快速生成 requiement.txt 命令
``` pip
pip freeze > requirements.txt
```
---
- RGB-> GRAY标准色彩转换方式
```
Gray = 0.299 · 𝑅 + 0.587 · 𝐺 + 0.114 · B
```
简化形式如下：
```
Gray = (𝑅 + 𝐺 + B) / 3
```