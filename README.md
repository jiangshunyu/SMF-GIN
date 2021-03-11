# SMF-GIN
  This is a Pytorch implementation for our submitted AI OPEN journal paper: Structure-Enhanced Meta-Learning For Few-Shot Graph Classification[arXiv]（http://arxiv.org/abs/2103.03547）
  
  Contributors: Shunyu Jiang, Fuli Feng, Weijian Chen, Xiangnan He
  
  ## Installation
  We used the following Python packages for core development. We tested on `Python 3.7`.
  ```
  pytorch                   1.0.1
  rdkit                     2019.03.1.0
  numpy                     
  json
  pandas
  ```
  
   ## Parameters

    + --lr: learning rate
    + --type: local-structure or global-structure
    + --norm_type： centering and scaling in paper
    + --attention_type: five attention models in paper
    + --num_ways: classes number in meta-task in paper
    + --spt_shots: support-set number in meta-task in paper
    + --qry_shots: query-set number in meta-task in paper
 
   ## Dataset
   We conduct experiments on the multi-class ***Chembl*** dataset and public dataset ***TRIANGLES***. Detailed information about these two datasets illustrated in paper.
   ***Chembl*** and ***TRIANGLES*** datasets were saved in ***'./dataset/dataset/'***.
   
   ## Examples
   We provide solutions of local structure and global structure respectively, which contain five different models of attention mechanism.
   
   ### TRIANGLES
   
   `nohup python main_local.py --type=local --attention_type=transformer --dataset=TRIANGLES --num_ways=3 > TRIANGLES_local_log.out &`
   
   or
   
   `nohup python main_global.py --type=gloabl --attention_type=self-attention --dataset=TRIANGLES --num_ways=3 > TRIANGLES_global_log.out &`
   
