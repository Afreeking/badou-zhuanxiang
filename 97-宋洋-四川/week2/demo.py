#编码：utf8 

进口  火炬 
进口  火炬 。 nn  作为  nn 
导入  numpy  为  np 
导入  随机 
导入  json 
导入  matplotlib 。 pyplot  作为  plt 

""" 
基于pytorch的网络编写 
实现一个网络完成一个简单nlp任务 
判断文本中是否有某些特定字符出现 
""" 

类  TorchModel ( nn . 模块 ): 
    def  __init__ ( self , vector_dim , sentence_length , vocab ): 
        超级 （ 火炬模型 ， 自我 ）。 __init__ () 
        自我 。 嵌入  =  nn 。 嵌入 ( len ( vocab ), vector_dim ) #embedding 层 
        自我 。 rnn  =  nn 。 RNN ( num_layers = 1 , input_size = vector_dim , hidden_​​size = vector_dim , batch_first = True ) #RNN 层 
        self . pool  =  nn . AvgPool1d ( sentence_length )    #池化层 
        self . classify  =  nn . Linear ( vector_dim ,  3 )      #线性层 
       
        self . loss  =  nn . functional . cross_entropy   # loss函数采用交叉熵 

    #当输入真实标签，返回loss值；无真实标签，返回预测值 
    def  forward ( self , x , y = None ): 
        x  =  自我 。 嵌入 ( x ) 
        _ , x  =  自我 。 rnn ( x ) 
        y_pred  =  自我 。 分类 ( x .squeeze ( )) 
        如果  y  不是  _  None ： 
            return  self . loss ( y_pred ,  y . squeeze ())    #预测值和真实值计算损失 
        否则 ： 
            return  y_pred                  #输出预测结果 

#字符集随便挑了一些字，实际上还可以扩充 
#为每个字生成一个标号 
#{"a":1, "b":2, "c":3...} 
#abc -> [1,2,3] 
def  build_vocab (): 
    chars  =  "abcdefghijklmnopqrstuvwxyz"   #字符集 
    vocab  =  {} 
    对于  index ， char  （  枚举 ： chars ） 
        vocab [ char ]  =  index    #每个字对应一个序号 
    词汇 [ 'unk' ] =  len ( 词汇 ) 
    返回  词汇 vocab

#随机生成一个样本 
#从所有字中选取sentence_length个字 
#反之为负样本 
def  build_sample ( vocab , sentence_length ): 
    #随机从字表选取sentence_length个字，可能重复 
    x  = [ 随机 。 选择 （ 列表 （ 词汇 。 键 （））） 为  _  的  范围内 （ 句子 ）] 
    #指定哪些字出现时为正样本 
    如果  设置 （ “abc” ） 和  设置 （ x ）： 
        y  =  0 
    #新增分类 
    elif  集 （ “xyz” ） 和  集 （ x ）： 
        y  =  1 
    #指定字都未出现，则为负样本 
    否则 ： 
        y  =  2 
    x  =  [ vocab . get ( word ,  vocab [ 'unk' ])  for  word  in  x ]    #将字转换成序号，为了做embedding 
    返回  x , y 

  
  #建立数据集 
#输入需要的样本数量。需要多少生成多少 
def  build_dataset ( sample_length , vocab , sentence_length ): 
    数据集_x  = [] 
    数据集_y  = [] 
    对于  i  的  范围内 ( sample_length )： 
        x , y  =  build_sample ( vocab , sentence_length ) 
        数据集_x 。 附加 ( x ) 
        数据集_y 。 附加 ([ y ]) 
    返回  火炬 。 LongTensor ( dataset_x )， 火炬 。 FloatTensor ( dataset_y ) 

#建立模型 
def  build_model ( vocab , char_dim , sentence_length ): 
    模型  =  TorchModel ( char_dim , sentence_length , vocab ) 
    返回  模型 

#测试代码 
#用来测试每轮模型的准确率 
def  评估 （ 模型 、 词汇 、 样本长度 ）： 
    模型 。 评估 （） 
    x ,  y  =  build_dataset ( 2000 ,  vocab ,  sample_length )    #建立200个用于测试的样本 
    print ( "本次预测集中共有%d个0类样本，%d个1类样本，%d个2类样本" % ( sum ( y . eq ( 0 )),  sum ( y . eq ( 1 )),  sum ( y . eq ( 2 )))) 
正确 , 错误  =  0 , 0 
    用  火炬 。 no_grad (): 
        y_pred  =  model ( x )       #模型预测 
        y_pred  =  火炬 。 argmax ( y_pred , dim = - 1 ) 
        正确  +=  int ( sum ( y_pred  ==  y .squeeze ( ))) 
        错误  +=  len ( y ) -  正确 
    print ( "正确预测个数：%d, 正确率：%f" % ( correct ,  correct / ( correct + wrong ))) 
    返回  正确 / （ 正确 + 错误 ） 


定义  主 （）： 
    #配置参数 
    epoch_num  =  200     #训练轮数 
    batch_size  =  20        #每次训练样本个数 
    train_sample  =  500     #每轮训练总共训练的样本总数 
    char_dim  =  20          #每个字的维度 
    sentence_length  =  6    #样本文本长度 
    learning_rate  =  0.005  #学习率 
    # 建立字表 
    vocab  =  build_vocab () 
    # 建立模型 
    模型  =  build_model ( 词汇 , char_dim , sentence_length ) 
    # 选择优化器 
    优化  =  火炬 。 优化 。 亚当 （ 模型 。 参数 （）， lr = learning_rate ） 
    日志  = [] 
    # 训练过程 
    对于  纪元  的  范围内 （ epoch_num ）： 
        模型 。 火车 () 
        watch_loss  = [] 
        对于  批次  的  范围内 （ int （ train_sample  /  batch_size ））： 
            x ,  y  =  build_dataset ( batch_size ,  vocab ,  sentence_length )  #构造一组训练样本 
            optim . zero_grad ()     #梯度归零 
            loss  =  model ( x ,  y )    #计算loss 
            loss . backward ()       #计算梯度 
            optim . step ()          #更新权重 
            watch_loss 。 追加 （ 损失 。 项目 （）） 
        print ( "========= \n 第%d轮平均loss:%f"  %  ( epoch  +  1 ,  np . mean ( watch_loss ))) 
        acc  =  evaluate ( model ,  vocab ,  sentence_length )    #测试本轮模型结果 
        日志 。 附加 ([ acc , np . mean ( watch_loss )]) 
    #画图 
    plt . plot ( range ( len ( log )), [ l [ 0 ]  for  l  in  log ],  label = "acc" )   #画acc曲线 
    plt . plot ( range ( len ( log )), [ l [ 1 ]  for  l  in  log ],  label = "loss" )   #画loss曲线 
    plt _ 传说 () 
    plt . show () 
    #保存模型 
    火炬 。 保存 （ 模型 。state_dict model.pth （）， ” ） 
    # 保存词表 
    作家  =  打开 （ “vocab.json” ， “w” ， 编码 = “utf8” ） 
    作家 。 写 （ json 。 转储 （ 词汇 ， ensure_ascii = False ， 缩进 = 2 ）） 
    作家 。 关闭 () 
    返回 

#使用训练好的模型做预测 
def  预测 ( model_path , vocab_path , input_strings ): 
    char_dim  =  20   # 每个字的维度 
    sentence_length  =  6   # 样本文本长度 
    vocab  =  json . load ( open ( vocab_path ,  "r" ,  encoding = "utf8" ))  #加载字符表 
    model  =  build_model ( vocab ,  char_dim ,  sentence_length )      #建立模型 
    model . load_state_dict ( torch . load ( model_path ))              #加载训练好的权重 
    x  = [] 
    对于  input_strings  中的  input_strings ： 
        x . append ([ vocab [ char ]  for  char  in  input_string ])   #将输入序列化 
    model . eval ()    #测试模式 
    with  torch . no_grad ():   #不计算梯度 
        result  =  model . forward ( torch . LongTensor ( x ))   #模型预测 
    对于  i ， input_string  中的  枚举 ( input_strings )： 
        print ( "输入：%s, 预测类别：%d, 概率值：%f"  %  ( input_string ,  round ( float ( result [ i ])),  result [ i ]))  #打印结果 



如果  __name__  ==  "__main__" ： 
    主要 （） 
    # test_strings = ["ffvaee", "cwsdfg", "rqwdyg", "nlkwww"] 
    # predict("model.pth", "vocab.json", test_strings) 