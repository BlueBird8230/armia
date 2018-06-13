## ARMIA建模分析

- [AR(I)MA时间序列建模过程——步骤和python代码](https://www.jianshu.com/p/cced6617b423)
- [数据分析系列（4）：基于ARMA模型的资金渠道流入流出预测](https://plushunter.github.io/2018/05/23/%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E7%B3%BB%E5%88%97%EF%BC%884%EF%BC%89%EF%BC%9A%E5%9F%BA%E4%BA%8EARMA%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%B5%84%E9%87%91%E6%B8%A0%E9%81%93%E6%B5%81%E5%85%A5%E6%B5%81%E5%87%BA%E9%A2%84%E6%B5%8B/)

### Usage
#### dependencies:
    1. python3.5+
    2. pip install virtualenv
    3. pip install -r requirements.txt
    4. enable env
        (unix-like: source env/bin/activate)
        (windows: .env/Scripts/activate.bat)
    5. python armia.py

#### 说明
    1. 原文的代码可读性很差，parse.py是一边读原文代码一边修改的结果，不过没写完。
    2. armia.py使用了pyflux和seaborn进行分析，数据集是data_armia.csv，
    其中diff那一列被去掉了，这样吻合原文代码的数据集。
    3. armia.py做的事情是：先将原序列取对数，再差分，对差分序列建模分析。预测值
    会被还原。

#### 注意
    * 原文件中best_diff方法疑似有一个bug，已修正，不过不影响计算结果，见注释：index取0而不是1
