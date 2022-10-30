# TwoTowersModel

基于`tensorflow2.9.0`实现了推荐系统中的基础双塔模型~~大部分是融合官方给出的代码~~。

强烈推荐官方文档[tensorflow recommender](https://www.tensorflow.org/recommenders)😏

## 主要的工作
1. 因实际使用需要，训练数据读入部分采用`csv`文件格式读入，然后转为`Tensor`作为模型的输入。代码中采用`pandas.read_csv`先将训练数据读入成`dataframe`类型再转为`Tensor`。

2. 由于实际使用中特征数量非常大，且不同的特征需要采用不同的处理方式；目前已实现配置这些特征的属性就可自动完成处理。目前支持:

    1. 离散类型特征为`string`类型，则采用[`StringLookup`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/StringLookup)完成编码
    2. 离散类型特征为`int`类型，则采用[`IntegerLookup`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/IntegerLookup)完成编码
    3. 连续类型特征若需要归一化，则采用[`Normalization`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization)完成
  



## TODO 
- [ ] 整理目前实现的功能，提交到Github
- [ ] 增加自动调参模块
