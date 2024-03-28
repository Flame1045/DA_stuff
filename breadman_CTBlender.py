class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator).__init__()
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1,
                  padding=0, bias=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1,
                               padding=0, bias=True)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1,
                               padding=0, bias=True)
        self.fc = nn.Linear(100, 1, bias=True)
        self.Alpha = torch.tensor(1.0, requires_grad=False)
        
    def forward(self,feat):
        
        
        x = self.grl(feat, self.Alpha)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(x)  
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)        

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            # alpha.retain_grad()
            # print("self.alpha grad:",alpha.grad)
            grad_input = - alpha*grad_output
        # print("grad_ouput",grad_output)
        # print("grad_input",grad_input)
        return grad_input, None

class GradientReversalM(nn.Module):
    def __init__(self):
        super().__init__()
        # self.alpha = torch.tensor(alpha, requires_grad=True)
        # self.alpha = alpha

    def forward(self, x, alpha):
        return GradientReversal.apply(x, alpha)
class SMFandSAF(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.num_groups = 16
        self.Channel_Shuffle = Channel_Shuffle(8)      
        self.ChannelWiseFusionModule = ChannelWiseFusionModule()
        self.SpatialWiseFusionModule = SpatialWiseFusionModule()  
        self.ScaleAggregationFusionModule = ScaleAggregationFusionModule()
    def forward(self, CNN_features, Transformer_features): 
        Transformer_features = Transformer_features.permute(0, 2, 1)        
        merge = []
        for k in range(len(CNN_features)):
            prepare_merge = []
            CNN_feature_current = CNN_features[k]
            # get tranformer feature 
            B, C, H, W = CNN_feature_current.size()
            Transformer_feature_current = Transformer_features[:, :, :H * W].view(B, C, H, W)
            Transformer_features = Transformer_features[:, :, H * W:]
            # make group 
            grouped_Transfomer = torch.chunk(Transformer_feature_current, self.num_groups, dim=1)
            grouped_CNN = torch.chunk(CNN_feature_current, self.num_groups, dim=1)
            # input group memeber
            for i in range(self.num_groups):
               
                SpatialWiseTensor = self.SpatialWiseFusionModule(grouped_Transfomer[i])
                x =  grouped_CNN[i] * SpatialWiseTensor # finish spatial attention
                
                ChannelWiseTensor = self.ChannelWiseFusionModule(grouped_Transfomer[i])
                ChannelWiseTensor = ChannelWiseTensor.expand(ChannelWiseTensor.size(0), ChannelWiseTensor.size(1), x.size(2), x.size(3))
                prepare_merge.append( x * ChannelWiseTensor )# finish channel attention
            
            merge.append(self.Channel_Shuffle(torch.cat(prepare_merge, dim =1)))
            
        tensor = self.ScaleAggregationFusionModule(merge)
        tensor[1]=nn.functional.interpolate(tensor[1],[tensor[0].size(2),tensor[0].size(3)],mode='nearest')
        tensor[2]=nn.functional.interpolate(tensor[2],[tensor[0].size(2),tensor[0].size(3)],mode='nearest')
        tensor[3]=nn.functional.interpolate(tensor[3],[tensor[0].size(2),tensor[0].size(3)],mode='nearest')
        output = tensor[0]+tensor[1]+tensor[2]+tensor[3]
            
            
        return output
class ScaleAggregationFusionModule(nn.Module):
    def __init__(self):
        super(ScaleAggregationFusionModule,self).__init__()  
        self.fc1 = nn.Linear(1,256)
    def forward(self,merge):
        for i,tensor in enumerate(merge):
            # GAP
            avgpool = F.adaptive_avg_pool2d(tensor, (1,1))
            # get alpha 
            alpha = torch.sum(avgpool, dim=1, keepdim=True)
            ### fully connected layer
            alpha = alpha.view(merge[i].size(0),-1)
            connected = self.fc1(alpha)
           
            connected = torch.unsqueeze(torch.unsqueeze(connected, dim=-1),dim = -1)
            
            over = connected*tensor
            
            merge[i] = over
        return merge
class ChannelWiseFusionModule(nn.Module):
    def __init__(self):
        super(ChannelWiseFusionModule,self).__init__()
        self.Sigmoid = nn.Sigmoid()
    def forward(self,x):
        B,C,H,W = x.size()
        self.LearnableMap = LearnableMap(B,C,H,W)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.LearnableMap(x)
        x = self.Sigmoid(x)
        return x 
class SpatialWiseFusionModule(nn.Module):
    def __init__(self):
        super(SpatialWiseFusionModule,self).__init__()    
        self.Sigmoid = nn.Sigmoid()

    def forward(self,x): #grouped Transformer featuere
        # normalization
        x = self.normalization(x)
        # Spatial fusion
        x = self.SpatialWiseFusion(x)
        # Sigmoid 
        x = self.Sigmoid(x)
        return x 
    def normalization(self,x):
        meanT = torch.mean(x, dim=(2, 3), keepdim=True)

        stdT = torch.std(x, dim=(2, 3), keepdim=True)

        tensor = (x - meanT) / stdT

        return tensor
    def SpatialWiseFusion(self,x):
        B,C,H,W = x.size()
        self.LearnableMap= LearnableMap(B,C,H,W)

        x = self.LearnableMap(x)
        return x 



class LearnableMap(nn.Module):
    def __init__(self,B,C,H,W):
        super(LearnableMap,self).__init__()
        self.w1 = nn.Parameter(torch.ones(B, C, H, W)).to("cuda:0")
        self.b1 = nn.Parameter(torch.zeros(B, C, H, W)).to("cuda:0")
    def forward(self,x ):
        x = self.w1 * x + self.b1
        return x 
    
class Channel_Shuffle(nn.Module):
    def __init__(self, num_groups):
        super(Channel_Shuffle,self).__init__()
        self.num_groups = num_groups
    def forward(self, x: torch.FloatTensor):
        batch_size, chs, h, w = x.shape
        chs_per_group = chs // self.num_groups
        x = torch.reshape(x, (batch_size, self.num_groups, chs_per_group, h, w))
        # (batch_size, num_groups, chs_per_group, h, w)
        x = x.transpose(1, 2)  # dim_1 and dim_2
        out = torch.reshape(x, (batch_size, -1, h, w))
        return out



class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=in_features, out_features=out_features,bias=True)

    def forward(self, x):
        return self.linear_layer(x)