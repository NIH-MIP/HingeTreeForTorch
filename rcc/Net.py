# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# March 2023
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import torch
import torch.nn as nn
import torch.nn.functional as F
from RandomHingeForest import RandomHingeForest

class Net(nn.Module):
    def __init__(self, in_channels, out_channels, numTrees = 100, depth = 7, use_expand=True):
        super(Net, self).__init__()

        self.bn11 = nn.BatchNorm2d(40, affine=False)
        self.conv11 = nn.Conv2d(in_channels, 40, 3, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(40, affine=False)
        self.conv12 = nn.Conv2d(40, 40, 3, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(40, affine=False)
        self.conv13 = nn.Conv2d(40, 40, 3, padding=1, bias=False)
    
        #self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
    
        self.bn21 = nn.BatchNorm2d(40, affine=False)
        self.conv21 = nn.Conv2d(40, 40, 3, padding=1, bias=False)
        self.bn22 = nn.BatchNorm2d(40, affine=False)
        self.conv22 = nn.Conv2d(40, 40, 3, padding=1, bias=False)
        self.bn23 = nn.BatchNorm2d(40, affine=False)
        self.conv23 = nn.Conv2d(40, 40, 3, padding=1, bias=False)
    
        self.bn31 = nn.BatchNorm2d(40, affine=False)
        self.conv31 = nn.Conv2d(40, 40, 3, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(40, affine=False)
        self.conv32 = nn.Conv2d(40, 40, 3, padding=1, bias=False)
        self.bn33 = nn.BatchNorm2d(40, affine=False)
        self.conv33 = nn.Conv2d(40, 40, 3, padding=1, bias=False)

    
        """
        self.conv41 = nn.Conv2d(40, 40, 3, padding=1, bias=False)
        self.conv42 = nn.Conv2d(40, 40, 3, padding=1, bias=False)
        self.conv43 = nn.Conv2d(40, 40, 3, padding=1, bias=False)
        """

        #self.features = nn.Linear(in_features=40, out_features=100, bias=False)
        self.forestbn1 = nn.BatchNorm2d(100, affine=False)
        self.features = nn.Conv2d(40,100,1, bias=False)
  
        self.forest = RandomHingeForest(in_channels=100, out_channels=numTrees, extra_outputs=[8,8], depth=depth)

        if use_expand:
            from HingeTree import Expand
            self.expand = Expand()
            print("Warning: Using HingeTree.Expand which is not compatible with the original model weights.", flush=True)
        else:
            self.expand = None
        #self.forest = nn.ConvTranspose2d(100, 100, 8, stride=8, output_padding=0)
        #self.agg = nn.Linear(in_features=numTrees, out_features=out_channels)
        self.agg = nn.Conv2d(numTrees,out_channels,1)
        #self.agg = nn.Conv2d(numTrees,out_channels,3, padding=1)

    def calculate_features(self, x):
        x = self.activation(self.bn11(self.conv11(x)))
        x = self.activation(self.bn12(self.conv12(x)))
        x = self.pool(self.activation(self.bn13(self.conv13(x))))

        x = self.activation(self.bn21(self.conv21(x)))
        x = self.activation(self.bn22(self.conv22(x)))
        x = self.pool(self.activation(self.bn23(self.conv23(x))))

        x = self.activation(self.bn31(self.conv31(x)))
        x = self.activation(self.bn32(self.conv32(x)))
        x = self.pool(self.activation(self.bn33(self.conv33(x))))

        """
        x = self.activation(self.conv41(x))
        x = self.activation(self.conv42(x))
        x = self.pool(self.activation(self.conv43(x)))
        """

        #x = x.view(-1, 40*2*2)
        x = self.features(x)

        return x

    def forward(self, x):
        origShape = x.shape

        x = self.calculate_features(x)

        x = self.forest(self.forestbn1(x))
        #x = F.normalize(x, p=2)


        if self.expand is None:
            # Original way used for paper

            # x is (B, out_channels, h, w, 8, 8)
            tmp = x.view([x.shape[0], x.shape[1], x.shape[2]*x.shape[3], x.shape[4]*x.shape[5]])
            # (B, out_channels, h*w, 8*8)
            tmp = tmp.transpose(2,3)
            # (B, out_channels, 8*8, h*w)
            tmp = tmp.transpose(1,2) # XXX: This makes Expand() not the same but was present in the original code
            # (B, 8*8, out_channels, h*w)
            tmp = tmp.reshape([x.shape[0], x.shape[1]*x.shape[4]*x.shape[5], x.shape[2]*x.shape[3]])
            # (B, 8*8*out_channels, h*w)

            x = F.fold(tmp, [x.shape[2]*x.shape[4], x.shape[3]*x.shape[5]], [x.shape[4], x.shape[5]], stride=[x.shape[4], x.shape[5]])

            """
            tmpInput = torch.eye(x.shape[2]*x.shape[3], dtype=x.dtype, device=x.device).view([1, x.shape[2]*x.shape[3], x.shape[2], x.shape[3]])
            tmpWeights = x.view([x.shape[0]*x.shape[1], x.shape[2]*x.shape[3], 2, 2]).transpose(0,1)

            x = F.conv_transpose2d(tmpInput, tmpWeights, stride=2).view([x.shape[0], x.shape[1], 2*x.shape[2], 2*x.shape[3]])
            """
        else:
            x = self.expand(x) # XXX: Due to a transpose() discrepancy, this is not the same as the original code


        #x = x.view(list(x.shape[:2]) + list(origShape[2:]))

        x = self.agg(x)
        
        #x = F.interpolate(x, size=origShape[2:], mode="nearest")

        return x

    def leafmap(self, x):
        origShape = x.shape

        x = self.calculate_features(x)

        x = self.forestbn1(x)
        #maps = x

        x = self.forest.leafmap(x)

        #print(maps[:,23,:,:][x[:,0,:,:] == 22].max())

        x = F.interpolate(x, size=origShape[2:], mode="nearest")

        return x

if __name__ == "__main__":
    net = Net(in_channels=4, out_channels=4, use_expand=True).cuda()

    x = torch.randn([8,4,256,256]).cuda()

    y_new = net(x)

    #print(y_new.shape)

    net.expand = None

    y = net(x)

    #print(y.shape)

    print((y - y_new).abs().max())
    exit()

    print(x.shape)
    print(y.shape)
    #print(x.max())
    #print(y.max())

