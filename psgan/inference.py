import torch
from PIL import Image

from .solver import Solver
from .preprocess import PreProcess


class Inference:
    """
    An inference wrapper for makeup transfer.
    It takes two image `source` and `reference` in,
    and transfers the makeup of reference to source.
    """
    def __init__(self, config, device="cpu", model_path="assets/models/G.pth"):
        """
        Args:
            device (str): Device type and index, such as "cpu" or "cuda:2".
            device_id (int): Specefying which devide index
                will be used for inference.
        """
        self.device = device
        self.solver = Solver(config, device, inference=model_path)
        self.preprocess = PreProcess(config, device)

    def transfer(self, source: Image, reference: Image, source_seg:Image,reference_seg:Image):
        """
        Args:
            source (Image): The image where makeup will be transfered to.
            reference (Image): Image containing targeted makeup.
        Return:
            Image: Transfered image.
        """
        source_input,real1 = self.preprocess(source,source_seg) #source_input=[mask,real]
        reference_input,real2 = self.preprocess(reference,reference_seg)
        
        result = self.solver.test(source_input, reference_input,real1, real2)

        return result
