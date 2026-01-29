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

import os
from RCCSeg import RCCSeg
from rcc_common import LoadImage, SaveImage, LoadMask, CleanUpMask

def main():
    #dataRoot="/data/AIR/RCC/Nifti"
    #dataRoot="/data/AIR/RCC/NiftiNew"
    #dataRoot="/data/AIR/RCC/NiftiNew"
    dataRoot="/scratch/cluster_scratch/layns/RCC/NiftiNew"
    #outputRoot="ProbMaps_UpdatedTrain_Requested4"
    #outputRoot="/data/AIR/layns/ProbMaps_unweighted_hingeforest_depth7_vggblock3_randomSplit1_epoch_549"
    outputRoot="/scratch/cluster_scratch/layns/RCC/Output"
    #outputRoot="ProbMaps_FourChannel_MoreTraining_Training"
    #testList=os.path.join(dataRoot, "valList.txt")
    #testList=os.path.join(dataRoot, "test_isbi2022_randomSplit2.txt")
    testList=os.path.join(dataRoot, "test_isbi2022_easyhard_randomSplit1.txt")
    #testList=os.path.join(dataRoot, "all.txt")
    #testList=os.path.join(dataRoot, "trainList.txt")
    #testList=os.path.join(dataRoot, "good.txt")
    #testList=os.path.join(dataRoot, "all.txt")

    cad = RCCSeg()
    cad.SetDevice("cuda:0")
    cad.SetDataRoot(dataRoot)
    #cad.LoadModel("snapshots_Unweighted/epoch_299.pt")
    #cad.LoadModel("snapshots/epoch_299.pt")
    #cad.LoadModel("snapshots_MaskOutCysts/epoch_294.pt")
    #cad.LoadModel("snapshots_MaskOutCystsLower/epoch_299.pt")
    #cad.LoadModel("snapshots_TwoChannel_MoreData/epoch_299.pt")
    #cad.LoadModel("snapshots_TwoChannel_MoreData_UpdatedTraining/epoch_299.pt")
    #cad.LoadModel("snapshots_SixChannels/epoch_299.pt")
    #cad.LoadModel("snapshots_SixChannels_MoreTraining/epoch_96.pt")
    #cad.LoadModel("snapshots_FourChannels_MoreTraining/epoch_491.pt")
    #cad.LoadModel("snapshots_FourChannels_MoreTraining/epoch_896.pt")
    #cad.LoadModel("snapshots_FourChannels_MoreTraining_CorrectedLabels/epoch_393.pt")
    #cad.LoadModel("snapshots_FourChannels_Weighted_RelabeledUnknowns/epoch_599.pt")
    #cad.LoadModel("/data/AIR/RCC/ISBI/snapshots_unweighted_randomSplit2/epoch_299.pt")
    #cad.LoadModel("/data/AIR/RCC/ISBI/snapshots_unweighted_easyhard_leaky_randomSplit7/epoch_249.pt")
    #cad.LoadModel("/data/AIR/RCC/ISBI/snapshotsHinge/epoch_249.pt")
    #cad.LoadModel("/data/AIR/RCC/ISBI/snapshots_weighted_easyhard_deterministic_hingeforest_randomSplit5/epoch_199.pt")
    #cad.LoadModel("/data/AIR/RCC/ISBI/snapshotsHinge_VGGBlock2_2x2Agg/epoch_97.pt")
    #cad.LoadModel("/data/AIR/RCC/ISBI/snapshotsHinge_VGGBlock3_8x8Agg_2/epoch_73.pt")
    #cad.LoadModel("/data/AIR/RCC/ISBI/snapshots_weighted_easyhard_deterministic_hingeforest_vggblock3_randomSplit4/epoch_749.pt")
    #cad.LoadModel("/data/AIR/RCC/ISBI/snapshotsHinge_VGGBlock_8x8_3/epoch_136.pt")
    cad.LoadModel("/scratch/cluster_scratch/layns/RCC/Models/rccseg_unweighted_easyhard_deterministic_depth7_hingeforest_vggblock3_randomSplit1/model.pt")
    
    if not os.path.exists(outputRoot):
        os.makedirs(outputRoot)

    caseList = []
    with open(testList, mode="rt", newline='') as f:
        caseList = [ line.strip() for line in f if len(line) > 0 ]

    print(caseList)
    #caseList = [ "0040-Subject-00159" ]

    for case in caseList:
        image, labelMap = cad.RunOne(case)

        labelMap = CleanUpMask(labelMap)

        newCase=case
        newCase = newCase.replace("/", "_")

        #outputPath=os.path.join(outputRoot, newCase + ".mha")
        #labelOutputPath=os.path.join(outputRoot, newCase +"_label.mha")
        outputPath=os.path.join(outputRoot, newCase + ".nii.gz")
        labelOutputPath=os.path.join(outputRoot, newCase +"_label.nii.gz")

        print(f"Info: Saving to '{outputPath}' ...")
        SaveImage(image, outputPath)
        SaveImage(labelMap, labelOutputPath)

if __name__ == "__main__":
    main()
