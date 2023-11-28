# RIPS_AMD_2020
 
1. Tensorboard Usage 

 - On your command prompt, type ssh -L 16006:127.0.0.1:6006 yourusername@the.server.you.run.model 
 - On your server command prompt, type tensorboard --logdir=path/to/log-directory. For example, tensorboard --logdir=runs/model_1
 - On your local machine, go to http://127.0.0.1:16006 to view the tensorboard display