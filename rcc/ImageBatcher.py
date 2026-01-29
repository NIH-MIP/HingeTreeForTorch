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
import random
import SimpleITK as sitk
import numpy as np
#from prostate_common import LoadDicomImage, LoadImage, SaveImage
from rcc_common import LoadImage, SaveImage, LoadMask, ShowWarnings
import torch
import queue
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor

class EndEpochToken:
    pass

class ImageBatcher:
    def __init__(self, dataRoot, listFile, batchSize, numClasses=4, seed=6112, dilateUnknown=False):
        ShowWarnings(False)
        self.dataRoot = dataRoot
        self.multipleOf = 16
        self.numChannels = 4
        self.batchSize = batchSize
        self.numClasses = numClasses
        self.dilateUnknown = dilateUnknown
        self.q = queue.Queue(maxsize=10)
        self.t = threading.Thread(target=self._loader_loop, args=(seed,))

        if isinstance(listFile, str):
            self._load_patient_ids(listFile)
        else:
            self.patientIds = listFile

    def start(self):
        if not self.t.is_alive():
            self.do_run = True
            self.t.start()

    def stop(self):
        while self.t.is_alive():
            self.do_run = False

            try:
                self.q.get_nowait()
            except:
                pass

            self.t.join(1.0)

    """
    def seed(self, value):
        random.seed(value)

    def shuffle(self):
        random.shuffle(self.patientIds)
    """

    def _loader_loop(self, seed):
        random.seed(seed)

        while self.do_run:
            random.shuffle(self.patientIds)

            for patientId in self.patientIds:
                if not self.do_run:
                    break

                volumePairs = self._load_patient(patientId)

                for volumePair in volumePairs:
                    if not self.do_run:
                        break

                    npVolume = volumePair[0]
                    npMask = volumePair[1]

                    npImageBatch = np.zeros([ self.batchSize ] + list(npVolume.shape[1:]), dtype=np.float32)
                    npMaskBatch = np.zeros([ self.batchSize ] + list(npMask.shape[1:]), dtype=np.int32)

                    zbegin = random.randint(0,self.batchSize-1)
                    for z in range(zbegin,npVolume.shape[0],self.batchSize):
                        if not self.do_run:
                            break

                        if z + self.batchSize <= npVolume.shape[0]:
                            begin = z
                            end = begin + self.batchSize

                            npImageBatch[...] = npVolume[begin:end, ...]
                            npMaskBatch[...] = npMask[begin:end, ...]
                        else:
                            begin = z
                            end = npVolume.shape[0]
                            offset = end-begin

                            npImageBatch[:offset, ...] = npVolume[begin:end, ...]
                            npMaskBatch[:offset, ...] = npMask[begin:end, ...]

                            begin = 0
                            end = self.batchSize - offset

                            npImageBatch[offset:, ...] = npVolume[begin:end, ...]
                            npMaskBatch[offset:, ...] = npMask[begin:end, ...]

                        imageBatch = torch.from_numpy(npImageBatch).type(torch.float32)
                        maskBatch = torch.from_numpy(npMaskBatch).type(torch.long)

                        #h = hashlib.md5()
                        #h.update(imageBatch.numpy().tobytes())
                        #h.update(maskBatch.numpy().tobytes())
                    
                        #self.q.put((imageBatch.clone(), maskBatch.clone(), h.hexdigest()))
                        self.q.put((imageBatch.clone(), maskBatch.clone()))

            self.q.put(EndEpochToken()) 


    def __iter__(self):
        return self

    def __next__(self):
        value = self.q.get()

        if isinstance(value, EndEpochToken):
            raise StopIteration

        return value

    def _load_patient_ids(self, listFile):
        self.patientIds = []

        with open(listFile, mode="rt", newline="") as f:
            self.patientIds = [ line.strip() for line in f if len(line.strip()) > 0 ]

    # Assume numpy convention
    def _get_roi_1d(self, size):
        remainder = (size % self.multipleOf)

        begin = int(remainder/2)
        end = begin + size - remainder

        return begin, end

    def _resize_image(self, npImg):
        beginX, endX = self._get_roi_1d(npImg.shape[-1])
        beginY, endY = self._get_roi_1d(npImg.shape[-2])

        return npImg[:,beginY:endY, beginX:endX].copy()

    def _load_patient(self, patientId):
        imageFiles = [ f"normalized{i+1}_aligned.nii.gz" if i > 0 else "normalized_aligned.nii.gz" for i in range(self.numChannels) ]
        imageFiles = [ os.path.join(self.dataRoot, "Images", patientId, imageFile) for imageFile in imageFiles ]

        maskFile = os.path.join(self.dataRoot, "Masks", patientId, "mask_aligned.nii.gz")

        with ThreadPoolExecutor(max_workers=min(self.numChannels, self.batchSize)) as exe:
            sitkVolumes = list(exe.map(LoadImage, imageFiles))


        assert all(sitkVolumes), patientId
        assert all([sitkVolumes[0].GetSize() == v.GetSize() for v in sitkVolumes])
        #sitkVolumes = [ LoadImage(imageFile) for imageFile in imageFiles ]

        #print([sitkVolume.GetPixelIDTypeAsString() for sitkVolume in sitkVolumes])
        #assert all([sitkVolume.GetPixelID() == sitk.sitkInt16 for sitkVolume in sitkVolumes])

        npVolumes = [ sitk.GetArrayViewFromImage(sitkVolume) for sitkVolume in sitkVolumes ]

        #assert all([ npVolume.min() < -150 for npVolume in npVolumes ])

        #if not all([npVolumes[0].shape == npVolume.shape for npVolume in npVolumes]):
        #    return

        sitkMask = LoadMask(maskFile, numClasses=self.numClasses, dilateUnknown=self.dilateUnknown)

        if not sitkMask:
            return

        npMask = sitk.GetArrayViewFromImage(sitkMask).astype(np.int32) # XXX: Prevent issue with -1

        assert npMask.shape == npVolumes[0].shape

        npMask[npMask == 255] = -1
        
        # TODO: Review this
        #npMask[np.logical_and(npVolumes[0] < 650, npMask == 1)] = -1
        npMask[np.logical_and(npVolumes[0] < 253, npMask == 1)] = -1

        for npVolume in npVolumes:
            #npMask[npVolume < 209] = -1
            npMask[npVolume < -188] = -1
        ######

        halfX = int(npVolumes[0].shape[-1]/2)

        npMaskRight = self._resize_image(npMask[:,:,:halfX])
        npMaskLeft = self._resize_image(npMask[:,:,halfX:])

        rightVoxelCount = (npMaskRight > 0).sum()
        leftVoxelCount = (npMaskLeft > 0).sum()
        maxVoxelCount = max(rightVoxelCount, leftVoxelCount)

        pairs = []

        sides=0
        #if npMaskRight.any():
        #if npMaskRight.max() > 0:
        if rightVoxelCount > 0.5*maxVoxelCount:
            sides += 1
            npVolumesRight = [ self._resize_image(npVolume[..., :halfX]) for npVolume in npVolumes ]
            npCombinedRight = np.zeros([ npVolumesRight[0].shape[0] ] + [ len(npVolumes) ] + list(npVolumesRight[0].shape[1:]), dtype=npVolumesRight[0].dtype)

            for c in range(len(npVolumesRight)):
                npCombinedRight[:,c, ...] = npVolumesRight[c][...]

            pairs.append((npCombinedRight.copy(), npMaskRight.copy()))
            pairs.append((npCombinedRight[..., ::-1].copy(), npMaskRight[..., ::-1].copy()))

        #if npMaskLeft.any():
        #if npMaskLeft.max() > 0:
        if leftVoxelCount > 0.5*maxVoxelCount:
            sides += 1
            npVolumesLeft = [ self._resize_image(npVolume[... ,halfX:]) for npVolume in npVolumes ]
            npCombinedLeft = np.zeros([ npVolumesLeft[0].shape[0] ] + [ len(npVolumes) ] + list(npVolumesLeft[0].shape[1:]), dtype=npVolumesLeft[0].dtype)

            for c in range(len(npVolumesLeft)):
                npCombinedLeft[:,c, ...] = npVolumesLeft[c][...]

            pairs.append((npCombinedLeft.copy(), npMaskLeft.copy()))
            pairs.append((npCombinedLeft[..., ::-1].copy(), npMaskLeft[... ,::-1].copy()))

        if sides != 1:
            print(sides, patientId)

        assert sides == 1
            
        return pairs

if __name__ == "__main__":
    dataRoot="/data/AIR/RCC/NiftiCombined"
    #dataRoot="/lscratch/38110568/NiftiCombined"
    listFile=os.path.join(dataRoot, "all_randomSplit1.txt")

    batcher = ImageBatcher(dataRoot, listFile, 32, numClasses=4, dilateUnknown=True)
    batcher.start()

    ShowWarnings(True)

    #batcher.seed(7271)

    for batch in batcher:
        imageBatch, maskBatch, digest = batch
        h = hashlib.md5()
        h.update(imageBatch.numpy().tobytes())
        h.update(maskBatch.numpy().tobytes())

        assert h.hexdigest() == digest

        print(f"{type(imageBatch)}, {type(maskBatch)}")
        print(f"{imageBatch.shape}, {maskBatch.shape}")

    batcher.stop()

