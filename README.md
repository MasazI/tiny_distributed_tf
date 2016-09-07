# tiny_distributed_tf
tiny_distributed_tf is a tiny implementaion of distributed tensorflow. Regression of airquality's data. 

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

---

Copyright (c) 2016 Masahiro Imai
Released under the MIT license
