# tiny_distributed_tf
tiny_distributed_tf is a tiny implementaion of distributed tensorflow. Regression of airquality's data. 

You can see a good introduction of distributed tensorflow in official site.  
https://www.tensorflow.org/versions/r0.10/how_tos/distributed/index.html

# Dependencies
- tensorflow (>=0.10)

# How to use
## locally run
- parameterserver index 0
```
python train.py --ps_hosts=localhost:2222,localhost:2223 --worker_hosts=localhost:2224,localhost:2225 --job_name=ps --task_index=0
```

- parameterserver index 1
```
python train.py --ps_hosts=localhost:2222,localhost:2223 --worker_hosts=localhost:2224,localhost:2225 --job_name=ps --task_index=1
```

- worker server index 1
```
python train.py --ps_hosts=localhost:2222,localhost:2223 --worker_hosts=localhost:2224,localhost:2225 --job_name=worker --task_index=1
```

- worker server index 0 as chief supervisor
```
python train.py --ps_hosts=localhost:2222,localhost:2223 --worker_hosts=localhost:2224,localhost:2225 --job_name=worker --task_index=0
```

# Others
This repository has fundamental codes of TensorFlow.
It might be useful to create distributed models.

- Scope, Session, Variables, Graph, Summary
- Linear Regression

---

Copyright (c) 2016 Masahiro Imai
Released under the MIT license
