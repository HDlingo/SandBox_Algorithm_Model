import torch


class DiscreateEncoder(torch.nn.Module):
  def __init__(self) -> None:
    super(DiscreateEncoder,self).__init__()
    # 预训练的bert
    
    self.type_nums = 10
    self.typedim = 96
    self.typeEmbedding = torch.nn.Embedding(self.type_nums, self.typedim)
    
    self.pos_dim= 96
    self.coordinateNum= 6
    self.coordinateEmbedding = self.positionEmbedding
    
    self.output_dim = 64
    self.linear = torch.nn.Linear(self.typedim + self.pos_dim, self.output_dim)
    
    
    pass
  
  def positionEmbedding(self, input_ls:torch.Tensor):
    # input_ls : (N,6)
    # expect output_dim % 6 ==0
    # return : (N, output_dim)
    # 6 = self.coordinateNum
    
    output_dim = self.pos_dim
    
    def cal_posi(x, i):
      # x : (N,1)
      return torch.sin(x / (10000 ** (2 * i / output_dim))),torch.cos(x / (10000 ** (2 * i / output_dim)))
     
    pos_encoding = torch.zeros((input_ls.size()[0] ,output_dim,))
    
    i_ls = list(range(0, output_dim // 6, 6))
    
    for i in range(6):
      for j in range(0, output_dim // 6, 2):
        pos_encoding[j+ i*output_dim // 6 , i], pos_encoding[j+1 +i*output_dim // 6 , i + 1] = cal_posi(input_ls[:,i] ,j)
        
    return pos_encoding
  
  def forward(self, input):
    
    input_coordinate_ls=[]
    input_type_ls = []
    
    posEmbed = self.positionEmbedding(input_coordinate_ls) # (N, dim)
    
    typeEmbed = self.typeEmbedding(input_type_ls) # (N, dim)
    
    feature = torch.concat([posEmbed, typeEmbed],dim=1) # (N, 2*dim)
    
    feature = self.linear(feature) # (N, output_dim)
    
    return feature
  